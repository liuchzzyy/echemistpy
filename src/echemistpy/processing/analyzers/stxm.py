# -*- coding: utf-8 -*-
"""STXM Data Analyzer for chemical mapping and spectroscopy analysis."""

from __future__ import annotations

import logging

import lmfit
import numpy as np
import scipy.linalg
import scipy.ndimage
import scipy.optimize
import xarray as xr
import umap
from scipy.cluster import vq
from skimage.registration import phase_cross_correlation
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from traitlets import Bool, Dict, Float, Int, List, Unicode

from echemistpy.io.structures import AnalysisData, AnalysisDataInfo, RawData
from echemistpy.processing.analyzers.registry import TechniqueAnalyzer

logger = logging.getLogger(__name__)


def b_value_model(x: np.ndarray, b: float, c: float) -> np.ndarray:
    """B-value model for thickness correction with improved numerical stability.

    Equation: I_thick = -ln((exp(-I_thin * C) + B) / (1 + B))
    Rewritten as: ln(1 + B) - ln(exp(-I_thin * C) + B)
    """
    # Enforce non-negative b for physical sense and stability during fitting exploration
    b_safe = max(b, 0.0)

    # Calculate terms
    # Term 1: ln(1 + B). Use log1p for precision when B is small
    term1 = np.log1p(b_safe)

    # Term 2: ln(exp(...) + B)
    # Prevent exp overflow/underflow (though x*c usually > 0, so exp -> 0)
    exp_val = np.exp(-x * c)

    # Add epsilon to prevent log(0) if exp_val -> 0 and b -> 0
    term2 = np.log(exp_val + b_safe + 1e-15)

    return term1 - term2


def build_lmfit_model(config: dict) -> tuple[lmfit.Model, lmfit.Parameters]:
    """Build a composite model from configuration dictionary using lmfit."""
    components = config.get("components", [])
    if not components:
        raise ValueError("No components defined for model")

    composite_model = None
    params = lmfit.Parameters()

    for i, comp in enumerate(components):
        ctype = comp["type"].lower()
        prefix = f"c{i}_"

        # Select model component
        if ctype == "gaussian":
            model = lmfit.models.GaussianModel(prefix=prefix)
        elif ctype == "lorentzian":
            model = lmfit.models.LorentzianModel(prefix=prefix)
        elif ctype == "linear":
            model = lmfit.models.LinearModel(prefix=prefix)
        elif ctype == "arctan" or ctype == "step":
            model = lmfit.models.StepModel(prefix=prefix, form="arctan")
        else:
            logger.warning(f"Unknown component type: {ctype}, skipping")
            continue

        # Initialize Parameters
        # lmfit models usually hint/guess parameters, but we rely on explicit config
        # Mapping config params to lmfit param names
        # Gaussian/Lorentzian: amplitude, center, sigma
        # Linear: slope, intercept
        # Step: amplitude, center, sigma

        comp_params = comp.get("params", {})
        bounds = comp.get("bounds", {})  # {"lower": [val1, val2...], "upper": ...}
        # Note: bounds format in previous implementation was list based on fixed param order.
        # lmfit uses named parameters. We need to adapt if we want backward compatibility
        # or expect user to provide dict of bounds per parameter name.
        # Let's support a more explicit config style: "params": {"center": {"value": 640, "min": 638, "max": 642}}
        # But for backward compat with my previous `test_stxm.py`, I used:
        # "params": {"center": 640.0, ...}, "bounds": {"lower": [...], "upper": [...]}
        # The list order was implicit. This is brittle.
        # Let's assume the user now provides explicit dict-based bounds if using lmfit,
        # OR we try to map the list if we know the order.

        # Standard lmfit param names
        model_param_names = model.param_names

        # Make default guesses to initialize parameters object
        model_params = model.make_params()

        # Update values from config
        for pname, pval in comp_params.items():
            full_name = f"{prefix}{pname}"
            if full_name in model_params:
                model_params[full_name].set(value=pval)
            else:
                # Handle aliases? e.g. "width" -> "sigma" for step?
                pass

        # If the user provided the "old style" bounds list, we try to apply them in standard order?
        # Gaussian/Lorentzian order in lmfit: amplitude, center, sigma
        # My previous test code used order: amplitude, center, sigma.
        # Linear: slope, intercept

        # If "bounds" key exists and has "lower"/"upper" lists
        lower_bounds = bounds.get("lower")
        upper_bounds = bounds.get("upper")

        # Define order map for supported types
        param_order = []
        if ctype in ["gaussian", "lorentzian"]:
            param_order = ["amplitude", "center", "sigma"]
        elif ctype in ["arctan", "step"]:
            param_order = ["amplitude", "center", "sigma"]  # StepModel uses sigma for width
        elif ctype == "linear":
            param_order = ["slope", "intercept"]

        if lower_bounds and len(lower_bounds) == len(param_order):
            for pname, lb in zip(param_order, lower_bounds):
                full_name = f"{prefix}{pname}"
                if full_name in model_params:
                    model_params[full_name].set(min=lb)

        if upper_bounds and len(upper_bounds) == len(param_order):
            for pname, ub in zip(param_order, upper_bounds):
                full_name = f"{prefix}{pname}"
                if full_name in model_params:
                    model_params[full_name].set(max=ub)

        # Add to composite
        if composite_model is None:
            composite_model = model
        else:
            composite_model += model

        params.update(model_params)

    return composite_model, params


class STXMAnalyzer(TechniqueAnalyzer):
    """Analyzer for Scanning Transmission X-ray Microscopy (STXM) data.

    Workflow:
    1. Preprocessing (Energy interpolation)
    2. Denoising (PCA)
    3. Background Removal (Pre-edge linear subtraction)
    4. Thickness Correction (B-Value method)
    5. Chemical Analysis (ROI Mapping, Onset Energy, Clustering)
    """

    technique = Unicode("txm")
    name = Unicode("STXMAnalyzer")

    # Configuration traits
    energy_step = Float(0.1, help="Energy step for interpolation (eV)")
    align_images = Bool(True, help="Perform image alignment during preprocessing")
    alignment_method = Unicode("phase_correlation", help="Method used for image alignment")
    alignment_upsample_factor = Int(10, help="Upsample factor for subpixel alignment precision")

    pca_components = Int(5, help="Number of PCA components for denoising")

    # UMAP configuration
    use_umap = Bool(False, help="Enable UMAP dimensionality reduction")
    umap_n_components = Int(2, help="Dimension of the embedded space (typically 2 for visualization)")
    umap_n_neighbors = Int(15, help="Number of neighbors for UMAP")
    umap_min_dist = Float(0.1, help="Minimum distance between points in embedding")
    umap_metric = Unicode("euclidean", help="Metric to use for UMAP")

    pre_edge_range = List(Float(), default_value=[625.0, 635.0], help="Pre-edge energy range for background removal (eV)")
    roi_maps = Dict(key_trait=Unicode(), value_trait=List(Float()), help="Dictionary of ROI definitions: {name: [start, end]}", default_value={})
    spatial_rois = Dict(key_trait=Unicode(), value_trait=List(Float()), help="Dictionary of Spatial ROI definitions: {name: [x_start, x_end, y_start, y_end]}", default_value={})
    roi_ranges = List(help="Legacy: List of (start, end) tuples. Use roi_maps instead.")
    clustering_method = Unicode("kmeans", help="Clustering algorithm: kmeans, minibatch_kmeans, gmm, dbscan")
    clustering_params = Dict(help="Additional parameters for clustering algorithm", default_value={})
    n_clusters = Int(3, help="Number of clusters for K-means segmentation")

    # Model Fitting configuration
    fitting_models = Dict(
        key_trait=Unicode(), value_trait=Dict(), help="Dictionary of model configurations for fitting spectra. Key is name, Value is dict with 'components', 'ranges', 'targets'", default_value={}
    )

    @property
    def required_columns(self) -> tuple[str, ...]:
        return ("optical_density",)

    def _align_stack(self, ds: xr.Dataset) -> xr.Dataset:
        """Align image stack to correct for drift."""
        if "optical_density" not in ds:
            return ds

        da = ds["optical_density"]
        # Ensure energy is first dimension for iterating
        if "energy" in da.dims:
            # transpose returns a new DataArray, doesn't modify ds in place yet
            da_aligned = da.transpose("energy", ...)
        else:
            return ds

        # Work on numpy array
        data = da_aligned.values.copy()
        n_images = data.shape[0]
        if n_images < 2:
            return ds

        # Reference: Middle image
        ref_idx = n_images // 2
        ref_img = np.nan_to_num(data[ref_idx])

        shifts = []
        logger.info("Aligning stack to frame %d...", ref_idx)

        for i in range(n_images):
            if i == ref_idx:
                shifts.append((0.0, 0.0))
                continue

            curr_img = np.nan_to_num(data[i])

            dy, dx = 0.0, 0.0
            if self.alignment_method == "phase_correlation":
                try:
                    # skimage.registration.phase_cross_correlation returns (shift, error, diffphase)
                    # We only need shift. upsample_factor enables subpixel precision.
                    shift, _, _ = phase_cross_correlation(ref_img, curr_img, upsample_factor=self.alignment_upsample_factor)
                    # Shift returned is (y, x) needed to align ref to moving (or vice versa depending on definition)
                    # Documentation says: "The shift vector (in pixels) required to register moving_image with reference_image."
                    # So moving + shift = reference.
                    # We want to shift moving to reference, so we apply the shift.
                    dy, dx = float(shift[0]), float(shift[1])
                except Exception as e:
                    logger.debug("Alignment failed for frame %d: %s", i, e)

            elif self.alignment_method == "center_of_mass":
                try:
                    # Calculate center of mass for reference and current image
                    # Invert images if features are dark on bright background?
                    # Usually OD is features bright on dark (0) background.
                    # If using transmission, features are dark.
                    # Assuming we are working on 'optical_density' (OD), features are bright (high OD).
                    cy_ref, cx_ref = scipy.ndimage.center_of_mass(ref_img)
                    cy_curr, cx_curr = scipy.ndimage.center_of_mass(curr_img)

                    dy = cy_ref - cy_curr
                    dx = cx_ref - cx_curr
                except Exception as e:
                    logger.debug("Center of mass alignment failed for frame %d: %s", i, e)

            shifts.append((dy, dx))

            # Apply shift
            # Use spline interpolation (order=1 for speed/stability)
            data[i] = scipy.ndimage.shift(data[i], (dy, dx), order=1, mode="nearest")

        # Update dataset
        # We assign back to the transposed shape.
        # If we assign to ds["optical_density"], xarray usually handles alignment by coords.
        # But 'data' is numpy array. We should wrap it in DataArray with correct dims.
        ds["optical_density"] = (da_aligned.dims, data)
        ds.attrs["alignment_shifts"] = shifts

        return ds

    def preprocess(self, raw_data: RawData) -> RawData:
        """Interpolate energy axis to uniform grid."""
        ds = raw_data.data

        # 0. Image Alignment (if enabled)
        if self.align_images:
            try:
                ds = self._align_stack(ds)
            except Exception as e:
                logger.warning("Image alignment failed: %s", e)

        if "energy" not in ds.coords:
            logger.warning("No 'energy' coordinate found. Skipping interpolation.")
            return raw_data

        energy = ds.coords["energy"].values
        if len(energy) < 2:
            return raw_data

        # Handle duplicate energy values if any
        if not ds.indexes["energy"].is_unique:
            logger.warning("Duplicate energy values found. averaging duplicates.")
            ds = ds.drop_duplicates("energy")
            energy = ds.coords["energy"].values

        # Create uniform energy grid
        e_min, e_max = energy.min(), energy.max()
        new_energy = np.arange(e_min, e_max + self.energy_step, self.energy_step)

        # Interpolate
        # We assume 'energy' is a dimension of 'optical_density'
        # xarray's interp handles missing dimensions gracefully (e.g. if rotation_angle depends on energy)
        cleaned_ds = ds.interp(energy=new_energy, method="linear", kwargs={"fill_value": "extrapolate"})

        return RawData(data=cleaned_ds)

    def compute(self, raw_data: RawData) -> tuple[AnalysisData, AnalysisDataInfo]:
        """Execute the STXM analysis workflow."""
        ds = raw_data.data.copy(deep=True)
        results = {}
        params_dict = {}

        # 1. Denoising (PCA)
        # Reshape to (Energy, Pixels)
        da = ds["optical_density"]
        # Ensure dimensions order
        if "energy" in da.dims and da.dims[0] != "energy":
            da = da.transpose("energy", ...)

        original_shape = da.shape
        n_energy = original_shape[0]
        # Flatten spatial dimensions
        flat_data = da.values.reshape(n_energy, -1)

        # Handle NaNs: replace with 0 or mean for PCA
        mask_nan = np.isnan(flat_data)
        flat_data_clean = np.nan_to_num(flat_data) if mask_nan.any() else flat_data

        # SVD (using sklearn PCA)
        try:
            # Flatten spatial dimensions for PCA: (n_samples, n_features)
            # In spectral imaging, typically we treat pixels as samples and energy as features?
            # Or Energy as samples (variables) and pixels as observations?
            # HyperSpy MVA typically treats the spectral axis as features (variables) and spatial pixels as observations.
            # But here flat_data is (n_energy, n_pixels).
            # So flat_data.T is (n_pixels, n_energy).

            X = flat_data_clean.T  # (n_samples=pixels, n_features=energy)

            # Use sklearn PCA
            n_comp = min(self.pca_components, min(X.shape))
            pca = PCA(n_components=n_comp)

            # Fit and Transform
            X_transformed = pca.fit_transform(X)  # (n_pixels, n_components)

            # Reconstruct: X_approx = X_transformed @ components_ + mean_
            X_reconstructed = pca.inverse_transform(X_transformed)  # (n_pixels, n_energy)

            # Transpose back to (n_energy, n_pixels)
            reconstructed_flat = X_reconstructed.T

            # Restore NaNs (if any were masked out, but here we filled them)
            if mask_nan.any():
                reconstructed_flat[mask_nan] = np.nan

            # Reshape back
            denoised_data = reconstructed_flat.reshape(original_shape)

            # Store in dataset
            ds["denoised"] = (da.dims, denoised_data)
            params_dict["pca_components"] = n_comp
            results["pca_explained_variance_ratio"] = pca.explained_variance_ratio_
            results["pca_singular_values"] = pca.singular_values_
            results["pca_components_matrix"] = pca.components_  # (n_components, n_features)

        except Exception as e:
            logger.error("PCA failed: %s", e)
            ds["denoised"] = da  # Fallback to original

        # Work with denoised data from here
        work_da = ds["denoised"]

        # 2. Background Removal
        # Fit linear background to pre-edge
        e_coords = work_da.coords["energy"].values
        mask_pre = (e_coords >= self.pre_edge_range[0]) & (e_coords <= self.pre_edge_range[1])

        if mask_pre.any():
            # Mean spectrum for pre-edge
            # We fit each pixel individually? Or just an offset?
            # Usually powerlaw or linear fit per pixel. For performance in python without numba,
            # fitting per pixel with curve_fit is slow.
            # Let's try a vectorized linear fit: y = mx + c
            # X matrix: [energy, 1]
            x_pre = e_coords[mask_pre]
            y_pre = work_da.isel(energy=mask_pre).values.reshape(len(x_pre), -1)

            # Linear regression: beta = (X^T X)^-1 X^T Y
            # X = column stack of (x_pre, ones)
            x_mat = np.vstack([x_pre, np.ones(len(x_pre))]).T
            # beta shape: (2, n_pixels) -> [slope, intercept]
            # Handle NaNs in Y_pre
            try:
                # Use least squares
                beta, _, _, _ = scipy.linalg.lstsq(x_mat, y_pre)

                # Calculate background for all energies
                x_full = np.vstack([e_coords, np.ones(len(e_coords))]).T
                background_flat = x_full @ beta
                background = background_flat.reshape(original_shape)

                ds["background_removed"] = work_da - background
            except Exception as e:
                logger.warning("Background removal failed: %s", e)
                ds["background_removed"] = work_da
        else:
            ds["background_removed"] = work_da

        work_da = ds["background_removed"]

        # 3. Thickness Correction (B-Value)
        # Simplified implementation:
        # 1. Calculate sum image (or max) to find thin/thick regions
        intensity_map = work_da.sum(dim="energy")
        # Flatten for percentile calculation
        flat_int = intensity_map.values.flatten()
        flat_int = flat_int[~np.isnan(flat_int)]

        if len(flat_int) > 0:
            p_low, p_high = np.percentile(flat_int, [10, 90])

            mask_thin = intensity_map < p_low
            mask_thick = intensity_map > p_high

            # Get mean spectra
            if mask_thin.any() and mask_thick.any():
                spec_thin = work_da.where(mask_thin).mean(dim=["x", "y"]).values
                spec_thick = work_da.where(mask_thick).mean(dim=["x", "y"]).values

                # Fit B-value model
                try:
                    popt, _ = scipy.optimize.curve_fit(b_value_model, spec_thin, spec_thick, p0=[0.1, 1.0], bounds=([0, 0], [np.inf, np.inf]))
                    b_val, c_val = popt
                    params_dict["b_value"] = float(b_val)
                    params_dict["c_value"] = float(c_val)

                    # Apply correction: I_corr = -ln((1+b)*exp(-I) - b)
                    # Note: The inverse of the B-value model I_thick = f(I_thin) is getting I_thin from I_thick (measured)
                    # Let y = I_thick (measured), x = I_thin (corrected/ideal)
                    # exp(-y) = (exp(-x*c) + b) / (1+b)
                    # exp(-x*c) = exp(-y)*(1+b) - b
                    # -x*c = ln(exp(-y)*(1+b) - b)
                    # x = -1/c * ln((1+b)*exp(-y) - b)

                    # Usually C is scaling factor, if we want true absorbance we might ignore C or set C=1 depending on definition.
                    # The notebook uses: data = -np.log((1 + popt_mapping[0]) * np.exp(-data) - popt_mapping[0])
                    # This corresponds to c=1 in the inverse formula above.

                    # Compute argument for log
                    arg = (1 + b_val) * np.exp(-work_da) - b_val
                    # Clip to avoid invalid log
                    arg = xr.where(arg > 1e-9, arg, 1e-9)
                    ds["thickness_corrected"] = -np.log(arg)

                except Exception as e:
                    logger.warning("B-Value correction failed: %s", e)
                    ds["thickness_corrected"] = work_da
            else:
                ds["thickness_corrected"] = work_da
        else:
            ds["thickness_corrected"] = work_da

        work_da = ds["thickness_corrected"]

        # 4. Chemical Analysis

        # Onset Energy (e.g. at 10% of max intensity)
        # We find the energy index where intensity crosses threshold
        # max_int = work_da.max(dim="energy")
        # threshold = 0.1 * max_int
        # TODO: Implement Onset Energy calculation properly

        # ROI Mapping
        # Support both legacy list and new dict
        rois_to_process = {}

        # Legacy list
        if self.roi_ranges:
            logger.warning("roi_ranges is deprecated, use roi_maps instead.")
            for start, end in self.roi_ranges:
                name = f"roi_{start}_{end}"
                rois_to_process[name] = (start, end)

        # New dict (overrides legacy if same name, but unlikely collision)
        if self.roi_maps:
            for name, rng in self.roi_maps.items():
                if len(rng) >= 2:
                    rois_to_process[name] = (rng[0], rng[1])

        for name, (start, end) in rois_to_process.items():
            # Ensure order
            s, e = sorted([start, end])
            mask_roi = (work_da.energy >= s) & (work_da.energy <= e)
            if mask_roi.any():
                roi_map = work_da.sel(energy=slice(s, e)).mean(dim="energy")
                ds[name] = roi_map
            else:
                logger.warning(f"ROI {name} ({s}-{e}) empty or out of range.")

        # Spatial ROIs (Extract spectra from regions)
        if self.spatial_rois:
            for name, coords in self.spatial_rois.items():
                if len(coords) >= 4:
                    x1, x2, y1, y2 = sorted(coords[0:2]) + sorted(coords[2:4])
                    # Assuming coords are indices/pixels or match coordinate values
                    # If x/y are integers (pixels), use isel (or sel if coords match).
                    # Usually users provide coordinates. If x/y are not monotonic, sel(slice) might fail.
                    # Here we assume coordinates are monotonic (standard image).
                    try:
                        # Use sel for coordinate based slicing
                        # Note: slice(min, max) includes bounds in xarray sel
                        roi_spec = work_da.sel(x=slice(x1, x2), y=slice(y1, y2)).mean(dim=["x", "y"])
                        ds[f"spectrum_{name}"] = roi_spec
                    except Exception as e:
                        logger.warning(f"Spatial ROI {name} processing failed: {e}")

        # Clustering
        # Use denoised or corrected data (here we use thickness_corrected)
        # Flatten spatial dims
        flat_data_cluster = work_da.values.reshape(n_energy, -1).T  # (n_pixels, n_energy)

        # Remove NaNs for clustering
        valid_pixels = ~np.isnan(flat_data_cluster).any(axis=1)
        data_for_clustering = flat_data_cluster[valid_pixels]

        # Optionally use PCA components for clustering if available?
        # Usually clustering on PCA scores is faster and cleaner.
        # But let's stick to full spectrum for generic approach, unless user asks.

        # UMAP Embedding
        if self.use_umap:
            try:
                # Use data_for_clustering (which is valid pixels from corrected/denoised data)
                # UMAP expects (n_samples, n_features) -> (pixels, energy)
                reducer = umap.UMAP(
                    n_components=self.umap_n_components,
                    n_neighbors=self.umap_n_neighbors,
                    min_dist=self.umap_min_dist,
                    metric=self.umap_metric,
                    random_state=42,  # Fix random state for reproducibility
                )
                embedding = reducer.fit_transform(data_for_clustering)

                # Reconstruct embedding maps (y, x, n_components)
                # We need to fill back the valid pixels
                # embedding shape: (n_valid_pixels, umap_n_components)

                # Create container for full image
                # original_shape is (energy, y, x)
                # map shape is (y, x)
                ny, nx = original_shape[1], original_shape[2]

                full_embedding = np.full((ny, nx, self.umap_n_components), np.nan)

                # Assign values
                # We need indices of valid pixels.
                # flat_data_cluster was reshaped from (n_energy, n_pixels) -> (n_pixels, n_energy)
                # valid_pixels is boolean mask of length n_pixels (ny*nx)

                # Flatten spatial dims of target
                full_flat = full_embedding.reshape(-1, self.umap_n_components)
                full_flat[valid_pixels] = embedding
                full_embedding = full_flat.reshape(ny, nx, self.umap_n_components)

                # Store in dataset
                # xarray dims: (y, x, umap_component)
                ds["umap_embeddings"] = (("y", "x", "umap_component"), full_embedding)

            except Exception as e:
                logger.warning("UMAP embedding failed: %s", e)

        if len(data_for_clustering) > self.n_clusters:
            try:
                labels_valid = None
                centroids = None

                # Dispatcher for sklearn methods
                method = self.clustering_method.lower()
                kwargs = self.clustering_params.copy()

                if method == "kmeans":
                    # n_clusters from trait or kwargs
                    n_c = kwargs.pop("n_clusters", self.n_clusters)
                    model = KMeans(n_clusters=n_c, **kwargs)
                    labels_valid = model.fit_predict(data_for_clustering)
                    centroids = model.cluster_centers_

                elif method == "minibatch_kmeans":
                    n_c = kwargs.pop("n_clusters", self.n_clusters)
                    model = MiniBatchKMeans(n_clusters=n_c, **kwargs)
                    labels_valid = model.fit_predict(data_for_clustering)
                    centroids = model.cluster_centers_

                elif method == "gmm":
                    n_c = kwargs.pop("n_components", self.n_clusters)
                    model = GaussianMixture(n_components=n_c, **kwargs)
                    labels_valid = model.fit_predict(data_for_clustering)
                    centroids = model.means_
                    # GMM also provides probabilities, maybe store them?

                elif method == "dbscan":
                    # DBSCAN doesn't use n_clusters
                    model = DBSCAN(**kwargs)
                    labels_valid = model.fit_predict(data_for_clustering)
                    # No centroids in DBSCAN strictly speaking, but we can compute mean of clusters
                    unique_labels = set(labels_valid)
                    centroids = []
                    for l in unique_labels:
                        if l != -1:
                            centroids.append(data_for_clustering[labels_valid == l].mean(axis=0))
                    centroids = np.array(centroids)

                else:
                    logger.warning(f"Unknown clustering method {method}, falling back to KMeans")
                    model = KMeans(n_clusters=self.n_clusters)
                    labels_valid = model.fit_predict(data_for_clustering)
                    centroids = model.cluster_centers_

                # Reconstruct full label map
                if labels_valid is not None:
                    full_labels = np.full(flat_data_cluster.shape[0], -1)
                    full_labels[valid_pixels] = labels_valid

                    label_map = full_labels.reshape(original_shape[1], original_shape[2])
                    ds["cluster_labels"] = (("y", "x"), label_map)

                    # Store centroids
                    if centroids is not None:
                        results["cluster_centroids"] = centroids

            except Exception as e:
                logger.warning("Clustering failed: %s", e)

        # Final cleanup (before fitting)

        # 5. Spectrum Fitting
        if self.fitting_models:
            fit_results = {}

            # Iterate over defined models
            for model_name, config in self.fitting_models.items():
                targets = config.get("targets", [])
                fit_range = config.get("range", None)

                try:
                    composite, params = build_lmfit_model(config)
                except ValueError as e:
                    logger.warning(f"Skipping model {model_name}: {e}")
                    continue

                if composite is None:
                    continue

                # Identify targets
                spectra_to_fit = {}

                if "cluster_centroids" in targets and "cluster_centroids" in results:
                    centroids = results["cluster_centroids"]
                    for i, centroid in enumerate(centroids):
                        spectra_to_fit[f"Cluster_{i}"] = (work_da.coords["energy"].values, centroid)

                for target in targets:
                    if target.startswith("spectrum_") and target in ds:
                        # Spatial ROI spectrum
                        da_spec = ds[target]
                        spectra_to_fit[target] = (da_spec.coords["energy"].values, da_spec.values)

                # Perform Fitting
                model_fits = {}
                for spec_name, (x, y) in spectra_to_fit.items():
                    # Filter range
                    if fit_range:
                        mask = (x >= fit_range[0]) & (x <= fit_range[1])
                        x_fit = x[mask]
                        y_fit = y[mask]
                    else:
                        x_fit, y_fit = x, y

                    if len(x_fit) == 0:
                        continue

                    try:
                        # lmfit fit
                        result = composite.fit(y_fit, params, x=x_fit)

                        # Store best fit params
                        # Extract params as list/dict?
                        # Store dict for flexibility: {name: value}
                        # Also store the ordered list values for backward compatibility if possible,
                        # but named is better.

                        # Flatten params to list of values matching component order?
                        # build_lmfit_model constructs components c0, c1...
                        # Param names: c0_amplitude, c0_center...
                        # Let's just store the result values in a structured way

                        # Extract parameter values in order of appearance in model?
                        # Or just a list of all variable parameters.
                        # For test compatibility, we need values.
                        # params order in result.params might be sorted? No, insertion order usually.

                        # Construct a list of parameter values that matches the "p0" expectation from before
                        # if we want to pass the existing test.
                        # Previous test expects: [c0_amp, c0_cen, c0_sigma, c1_slope, c1_intercept]
                        # Our prefixes are c0_, c1_.
                        ordered_values = []
                        for pname in result.params:
                            ordered_values.append(result.params[pname].value)

                        # Store
                        model_fits[spec_name] = {"params": ordered_values, "param_dict": result.best_values, "chisqr": result.chisqr, "fitted_curve": (x_fit, result.best_fit)}

                        # Save curve to dataset
                        # Evaluate on full x range
                        # composite.eval(params=result.params, x=x)
                        y_full_calc = composite.eval(result.params, x=x)

                        da_fit = xr.DataArray(y_full_calc, coords={"energy": x}, dims="energy", name=f"fit_{model_name}_{spec_name}")
                        ds[f"fit_{model_name}_{spec_name}"] = da_fit

                    except Exception as e:
                        logger.warning(f"Fitting {model_name} to {spec_name} failed: {e}")

                fit_results[model_name] = model_fits

            results["fitting_results"] = fit_results

        # Populate results
        analysis_data = AnalysisData(data=ds)

        # Populate info
        # params might be lmfit.Parameters which acts dict-like but could cause issues if passed directly
        # to something expecting a plain dict if not handled.
        # AnalysisDataInfo expects parameters to be a Dict.
        # If params is lmfit.Parameters, we should convert it.
        # But 'params' in compute() scope is actually 'params' dict initialized at start: params = {}
        # Wait, 'params' variable was shadowed?
        # In build_lmfit_model, it returns (model, params).
        # In compute(), I used: composite, model_params = build_lmfit_model(config)
        # Ah, I used: composite, params = build_lmfit_model(config)
        # This overwrote the 'params' dictionary I initialized at line ~203: params = {}
        # The 'params' passed to AnalysisDataInfo(parameters=params) is now an lmfit.Parameters object from the *last* model loop.
        # That is wrong. AnalysisDataInfo.parameters should store global analysis parameters (like pca_components).

        # Fix: rename the local variable in the loop.

        # Also need to fix the shadowing.

        analysis_info = AnalysisDataInfo(parameters=params_dict, others=results)

        return analysis_data, analysis_info
