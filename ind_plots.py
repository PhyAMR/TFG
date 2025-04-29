import logging
import os
import itertools
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dcor import distance_correlation
import hashlib
import pathlib

class DeduplicatedErrorHandler(logging.FileHandler):
    def __init__(self, filename):
        super().__init__(filename)
        self.error_cache = {}  # Key: (exception_type, exception_message), Value: {count, filename}

    def emit(self, record):
        if record.levelno < logging.ERROR:
            return

        msg = record.getMessage()
        exc_info = record.exc_info
        filename = None
        error_msg = msg

        # Extract filename from message
        if 'filename=' in msg:
            parts = msg.split('filename=', 1)
            if ';' in parts[1]:
                filename_part, error_part = parts[1].split(';', 1)
                filename = filename_part.strip()
                error_msg = error_part.strip()
            else:
                filename = parts[1].strip()

        # Determine exception type and message
        exc_type = None
        exc_message = None
        if exc_info and exc_info[0]:
            exc_type = exc_info[0].__name__
            exc_message = str(exc_info[1])
        else:
            exc_type = "Message"
            exc_message = error_msg

        key = (exc_type, exc_message)

        if key in self.error_cache:
            self.error_cache[key]['count'] += 1
            # Log reference
            ref_msg = (f"Error {exc_type}: {exc_message} occurred again "
                       f"(total {self.error_cache[key]['count']} times). "
                       f"First seen in: {self.error_cache[key]['filename']}")
            new_record = logging.makeLogRecord({
                'msg': ref_msg,
                'levelno': logging.ERROR,
                'name': record.name,
                'pathname': record.pathname,
                'lineno': record.lineno,
                'exc_info': None
            })
            super().emit(new_record)
        else:
            self.error_cache[key] = {'count': 1, 'filename': filename}
            # Log full error
            if filename:
                record.msg = f"filename={filename}; {error_msg}"
            super().emit(record)

# Configure logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Clear existing handlers
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Info file handler
info_handler = logging.FileHandler("plotting_info.log")
info_handler.setLevel(logging.INFO)
info_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
info_handler.setFormatter(info_formatter)

# Error file handler with deduplication
error_handler = DeduplicatedErrorHandler("plotting_errors.log")
error_handler.setLevel(logging.ERROR)
error_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
error_handler.setFormatter(error_formatter)

# Console handler
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(info_formatter)

root_logger.addHandler(info_handler)
root_logger.addHandler(error_handler)
root_logger.addHandler(stream_handler)

logger = logging.getLogger(__name__)

def sanitize_filename(name):
    """More aggressive sanitization"""
    name = str(name).strip()
    name = re.sub(r'[^a-zA-Z0-9_\-.]', '_', name)  # Only allow safe chars
    name = re.sub(r'_+', '_', name)  # Remove duplicate underscores
    return name[:100]  # Hard limit

# Read completed plots from checkpoint file
def load_completed_plots():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return set(line.strip() for line in f)
    return set()

# Save a completed plot identifier to the checkpoint file
def save_completed_plot(plot_id):
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(f"{plot_id}\n")


def plot_condition(name, unit_groups = {
        "Time_Plots": [
            "dex_yr",  
            "Gyr"
        ],
        "Magnitude_Plots": [
            "mag"       
        ],
        "StarFormation_Plots": [
            "solMass_yr"  
        ],
        "Metallicity_Plots": [
            "solMet"     
        ],
        "Luminosity_Plots": [
            "erg_s_Hz",  
            "dex_solLum"  
        ],
        "Mass_Plots": [
            "dex_solMass"  
        ]
    }):
    """
    Decides whether a plot should be processed or not based on its filename.
        name (str): Filename of the plot
        unit_groups (dict): Dictionary mapping folder names to lists of unit patterns
    Returns:
        True if the plot should be processed, False otherwise
    """

    for group_name, patterns in unit_groups.items():
        # Count how many patterns match components in the split filename
        same_matches = 0
        for pattern in patterns:
            same_matches += name.count(pattern)
        
        #print(f"Checking patterns in group '{group_name}'")
        #print(f"Number of same matches: {same_matches}")
        
        if same_matches >= 2:  # At least 2 distinct patterns must match
            logger.info(f"Plot {name} matches at least 2 patterns in group '{group_name}'")
            return False
        
    return True

def generate_centered_grid(data_x, data_y, n_points=50, std_scale=1.5):
    """Generate grid centered around data mean with std-scaled range"""
    x_mean = np.mean(data_x)
    x_std = np.std(data_x)
    y_mean = np.mean(data_y)
    y_std = np.std(data_y)
    
    return np.meshgrid(
        np.linspace(x_mean - std_scale*x_std, x_mean + std_scale*x_std, n_points),
        np.linspace(y_mean - std_scale*y_std, y_mean + std_scale*y_std, n_points)
    )


def plot2d(df, x, y, dire, n, heatmap_column, dataset_dir):
    """Generate 2D plots with all valid log/linear combinations."""
    try:
        logger.info(f"Creating 2D plot {n} for {x} vs {y}")
        
        # Check for valid logarithmic transformation candidates
        x_positive = not np.any(df[x] <= 0)
        y_positive = not np.any(df[y] <= 0)
        
        # Generate valid axis combinations
        axis_options = {
            'x': [False, True] if x_positive else [False],
            'y': [False, True] if y_positive else [False]
        }
        combinations = list(itertools.product(axis_options['x'], axis_options['y']))
        
        # Calculate correlations for all combinations
        correlations = {}
        for log_x, log_y in combinations:
            x_data = np.log10(df[x]) if log_x else df[x]
            y_data = np.log10(df[y]) if log_y else df[y]
            correlations[(log_x, log_y)] = distance_correlation(x_data, y_data)

        # Check if any correlation meets the threshold
        if max(correlations.values()) <= 0.7:
            logger.info(f"Skipping 2D plot {n} - all correlation coefficients are too low")
            return

        # Setup figure and axes
        safe_n = sanitize_filename(n)
        safe_x = sanitize_filename(x)
        safe_y = sanitize_filename(y)
        os.makedirs(dire, exist_ok=True)
        
        if safe_n in completed_plots:
            logger.info(f"Skipping plot {n}: already completed.")
            return

        # Create dynamic subplot grid
        n_plots = len(combinations)
        n_cols = min(n_plots, 2)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        sns.set_theme(style="whitegrid", font_scale=1.2)
        fig = plt.figure(figsize=(8 * n_cols, 6 * n_rows), dpi=300)
        plt.suptitle(f"{safe_x} vs {safe_y}", y=0.94, fontsize=16)
        
        # Adjust spacing for colorbar
        plt.subplots_adjust(
            left=0.1/n_cols,
            right=0.9,
            bottom=0.1,
            top=0.85,
            wspace=0.3,
            hspace=0.4
        )

        # Heatmap setup
        cmap = plt.get_cmap("viridis")
        norm = None
        if heatmap_column and heatmap_column in df:
            norm = plt.Normalize(df[heatmap_column].min(), df[heatmap_column].max())
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])

        # Plot each combination
        for idx, (log_x, log_y) in enumerate(combinations, 1):
            ax = fig.add_subplot(n_rows, n_cols, idx)
            
            # Transform data
            x_data = np.log10(df[x]) if log_x else df[x]
            y_data = np.log10(df[y]) if log_y else df[y]
            
            # Create labels
            x_label = f"log10({safe_x})" if log_x else safe_x
            y_label = f"log10({safe_y})" if log_y else safe_y
            
            # Plot points
            scatter = ax.scatter(
                x_data, y_data,
                c=df[heatmap_column] if heatmap_column else 'red',
                cmap=cmap, norm=norm,
                s=30, alpha=0.7
            )
            
            # Format labels
            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.set_title(f"DCOR: {correlations[(log_x, log_y)]:.2f}", fontsize=12)
            
            # Add grid
            ax.grid(True, alpha=0.3)

        # Add colorbar if using heatmap
        if heatmap_column and norm:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
            cbar = fig.colorbar(scatter, cax=cbar_ax)
            cbar.set_label(heatmap_column, fontsize=12, labelpad=15)
            cbar.ax.tick_params(labelsize=10)

        # Save plot
        filename = pathlib.Path(os.path.join(dire, f"2d_{safe_n}.png"))
        if len(str(filename)) > 100:
            logger.warning(f"Filename {filename} is too long, using hash instead")
            hash_id = hashlib.md5(safe_n.encode()).hexdigest()[:6]
            filename = pathlib.Path(os.path.join(dire, f"2d_{hash_id}.png"))
            
        plt.savefig(filename, bbox_inches="tight")
        logger.info(f"Saved 2D plot to {filename}")
        save_completed_plot(safe_n)

    except Exception as e:
        try:
            logging.error(f"Error in plot3d: {e}", exc_info=True)
        except Exception as log_err:
            print(f"Failed to log error: {log_err}")  # Fallback logging
        
        try:
            e_path = pathlib.Path(os.path.normpath(os.path.join(dataset_dir, "errors")))
            os.makedirs(e_path, exist_ok=True)
            hash_id = hashlib.md5(safe_n.encode()).hexdigest()[:6]
            filename = pathlib.Path(os.path.normpath(os.path.join(e_path, f"2d_{hash_id}.png")))
            plt.savefig(filename, bbox_inches="tight")
            logger.info(f"Saved 2D plot to {filename}")
            save_completed_plot(safe_n)
        except Exception as e:
            logger.error(f"Error in 2D plot {n}: {e}", exc_info=True)
    finally:
        plt.close('all')

def plot3d(df, x, y, z, dire, n, heatmap_column, dataset_dir):
    """Generate 3D plots with PCA analysis, save to specified directory."""
    fig = None
    try:
        logging.info(f"Starting 3D plot for {x}, {y}, {z} (n={n})")

        # Sanitize inputs and create directories
        safe_n = sanitize_filename(n)
        if safe_n in completed_plots:
            logger.info(f"Skipping plot {n}: already completed.")
            return
        base_dir = os.path.normpath(dire)
        dirs = {
            'good': os.path.join(base_dir, "g"),
            'regular': os.path.join(base_dir, "r"), 
            'trash': os.path.join(base_dir, "t"),
            'age': os.path.join(base_dir, "a")
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

        # Data preparation
        scope = df[[x, y, z]].apply(pd.to_numeric, errors='coerce').astype(np.float32)
        data_variants = []
        labels = []

        # Generate axis combinations
        if np.any(scope.values <= 0):
            # Identify which columns can be logarithmically transformed
            eligible_axes = [np.all(scope[col] > 0) for col in [x, y, z]]
            
            # Generate possible transformations only for eligible columns
            axis_options = []
            for col, can_log in zip([x, y, z], eligible_axes):
                axis_options.append([False, True] if can_log else [False])
            
            axis_combinations = list(itertools.product(*axis_options))
            
            data_variants = []
            labels = []
            for log_x, log_y, log_z in axis_combinations:
                temp_data = scope.copy()
                x_label = r"$\log_{10}$" + x if log_x else x
                y_label = r"$\log_{10}$" + y if log_y else y
                z_label = r"$\log_{10}$" + z if log_z else z
                
                if log_x: temp_data[x] = np.log10(temp_data[x])
                if log_y: temp_data[y] = np.log10(temp_data[y])
                if log_z: temp_data[z] = np.log10(temp_data[z])
                
                data_variants.append(temp_data)
                labels.append((x_label, y_label, z_label))
        else:
            # Original code for all-positive case
            axis_combinations = list(itertools.product([False, True], repeat=3))
            data_variants = []
            labels = []
            for log_x, log_y, log_z in axis_combinations:
                temp_data = scope.copy()
                x_label = r"$\log_{10}$" + x if log_x else x
                y_label = r"$\log_{10}$" + y if log_y else y
                z_label = r"$\log_{10}$" + z if log_z else z
                
                if log_x: temp_data[x] = np.log10(temp_data[x])
                if log_y: temp_data[y] = np.log10(temp_data[y])
                if log_z: temp_data[z] = np.log10(temp_data[z])
                
                data_variants.append(temp_data)
                labels.append((x_label, y_label, z_label))

        # Create figure with dynamic layout
        n_plots = len(data_variants)
        n_cols = min(n_plots, 4)
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig = plt.figure(figsize=(35, 10 * n_rows), dpi=300)  # Dynamic height based on rows

        # Create subplots with adjusted spacing
        axes = []
        for i in range(n_plots):
            axes.append(fig.add_subplot(n_rows, n_cols, i+1, projection='3d'))

        plt.subplots_adjust(left=0.05, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.3, hspace=0.2)

        # Color setup
        if heatmap_column and heatmap_column in df:
            cmap = plt.get_cmap("viridis")
            norm = plt.Normalize(df[heatmap_column].min(), df[heatmap_column].max())
            colors = cmap(norm(df[heatmap_column]))
        else:
            colors = 'red'

        # Track PCA results for all subplots
        pca_results = []

        # Plot each combination with adjusted label padding
        for ax, (data, (xl, yl, zl)) in zip(axes, zip(data_variants, labels)):
            # PCA analysis
            scaler = StandardScaler()
            points = scaler.fit_transform(data)
            pca = PCA(n_components=3).fit(points)
            pca_results.append(pca.explained_variance_ratio_[2])
            
            # Plot data with adjusted label positions
            scatter = ax.scatter(data[x], data[y], data[z], c=colors, s=8, alpha=0.6)
            ax.set_xlabel(xl, fontsize=10, labelpad=15)
            ax.set_ylabel(yl, fontsize=10, labelpad=15)
            ax.set_zlabel(zl, fontsize=10, labelpad=15)
            ax.tick_params(axis='both', which='major', labelsize=8, pad=8)
            
            # Add PCA plane if needed
            if pca.explained_variance_ratio_[2] <= 2e-1:
                xx, yy = generate_centered_grid(data[x], data[y])
                
                # Modified PCA plane equation using mean-centered coordinates
                xx_centered = xx - np.mean(data[x])
                yy_centered = yy - np.mean(data[y])
                zz = (-pca.components_[2][0] * xx_centered 
                    - pca.components_[2][1] * yy_centered) / pca.components_[2][2]
                zz += np.mean(data[z])
                ax.plot_surface(xx, yy, zz, alpha=0.25, color='gray')

                # Calculate view angles from PCA normal vector
                normal = pca.components_[2]
                nx, ny, nz = normal
                azimuth = abs(np.degrees(np.arctan2(ny, nx)))
                elevation = abs(np.degrees(np.arcsin(nz))/2)

                ax.view_init(elev=elevation, azim=azimuth)  # Set dynamic view
                ax.set_title(f'DCOR: {round(distance_correlation(data[[x,y]],data[z]),3)} PCA: {round(pca.explained_variance_ratio_[2], 3)}', 
                           fontsize=10, pad=5)


        # Add colorbar if needed
        if heatmap_column and heatmap_column in df:
            cbar_ax = fig.add_axes([0.15, 0.93, 0.7, 0.03])  # Centered horizontal bar
            cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
            cbar.set_label(heatmap_column, fontsize=16, labelpad=15)
            cbar.ax.tick_params(labelsize=12, rotation=45)
            cbar_ax.xaxis.set_ticks_position('top')
            cbar_ax.xaxis.set_label_position('top')
            cbar.outline.set_edgecolor('black')
            cbar.outline.set_linewidth(0.5)

        # Determine save location based on worst case
        max_variance = max(pca_results)
        if 'age' in safe_n.lower():
            save_dir = dirs['age']
            prefix = ""
        elif max_variance <= 1e-2:
            save_dir = dirs['good']
            prefix = ""
        elif max_variance <= 2e-1:
            save_dir = dirs['regular']
            prefix = ""
        else:
            save_dir = dirs['trash']
            prefix = ""
        
        # Save complete figure once
        filename = pathlib.Path(os.path.normpath(os.path.join(save_dir, f"3d_{safe_n}.png")))
        if len(str(filename)) > 100:
            logger.warning(f"Filename {filename} is too long, using hash instead")
            hash_id = hashlib.md5(safe_n.encode()).hexdigest()[:6]
            filename = pathlib.Path(os.path.normpath(os.path.join(save_dir, f"3d_{hash_id}.png")))
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        logging.info(f"Saved 3D plot to {filename}")
        save_completed_plot(safe_n)
    except Exception as e:
        try:
            logging.error(f"Error in plot3d: {e}", exc_info=True)
        except Exception as log_err:
            print(f"Failed to log error: {log_err}")  # Fallback logging
        
        try:
            e_path = pathlib.Path(os.path.join(dataset_dir, "errors"))
            os.makedirs(e_path, exist_ok=True)
            hash_id = hashlib.md5(safe_n.encode()).hexdigest()[:6]
            filename = pathlib.Path(os.path.normpath(os.path.join(e_path, f"3d_{hash_id}.png")))
            if fig is not None:
                plt.savefig(filename, bbox_inches="tight")
            logger.info(f"Saved 3D plot to {filename}")
            save_completed_plot(safe_n)
        except Exception as e:
            logger.error(f"Error in 3D plot {n}: {e}", exc_info=True)
    finally:
        plt.close('all')


if __name__ == "__main__":
    # Sample data
    physical_properties = 'Data/raw/hlsp_candels_hst_wfc3_egs_multi_v1_physpar-cat.txt'

    phys_prop_data = pd.read_csv(physical_properties, sep='\s+', header=None, skiprows=123)



    #A continuación añadimos las columnas

    phys_prop_data.columns = ["ID",
    "age_2a_tau (dex(yr))" ,
    "tau_2a_tau (Gyr)" ,
    "Av_2a_tau (mag)" ,
    "SFR_2a_tau (solMass/yr)" ,
    "chi2_2a_tau ",
    "age_4b (dex(yr))" ,
    "EBV_4b (mag)" ,
    "age_6a_tau (dex(yr))" ,
    "tau_6a_tau (Gyr)" ,
    "EBV_6a_tau (mag)" ,
    "SFR_6a_tau (solMass/yr)" ,
    "met_6a_tau (solMet)" ,
    "extlw_6a_tau ",
    "chi2_6a_tau ",
    "L1400_6a_tau (erg/s/Hz)" ,
    "L2700_6a_tau (erg/s/Hz)" ,
    "UMag_6a_tau (mag)" ,
    "BMag_6a_tau (mag)" ,
    "VMag_6a_tau (mag)" ,
    "RMag_6a_tau (mag)" ,
    "IMag_6a_tau (mag)" ,
    "JMag_6a_tau (mag)" ,
    "KMag_6a_tau (mag)" ,
    "age_10c (dex(yr))" ,
    "SFH_10c ",
    "tau_10c (Gyr)" ,
    "met_10c (solMet)" ,
    "M_l99_11a_tau (dex(solMass))",
    "M_u99_11a_tau (dex(solMass))",
    "age_11a_tau (dex(yr))" ,
    "SFR_11a_tau (solMass/yr)" ,
    "M_l68_12a (dex(solMass))",
    "M_u68_12a (dex(solMass))",
    "M_l95_12a (dex(solMass))",
    "M_u95_12a (dex(solMass))",
    "age_12a (dex(yr))" ,
    "tau_12a (Gyr)" ,
    "EBV_12a (mag)" ,
    "met_12a (solMet)" ,
    "Lbol_12a (dex(solLum))" ,
    "chi2_12a ",
    "age_13a_tau (dex(yr))" ,
    "tau_13a_tau (Gyr)" ,
    "Av_13a_tau (mag)" ,
    "SFR_13a_tau (solMass/yr)" ,
    "chi2_13a_tau ",
    "age_14a (dex(yr))" ,
    "SFH_14a ",
    "tau_14a (Gyr)" ,
    "EBV_14a (mag)" ,
    "SFR_14a (solMass/yr)" ,
    "q_14a ",
    "age_6a_tau ^NEB (dex(yr))" ,
    "tau_6a_tau ^NEB (Gyr)" ,
    "EBV_6a_tau ^NEB (mag)" ,
    "SFR_6a_tau ^NEB (solMass/yr)" ,
    "met_6a_tau ^NEB (solMet)" ,
    "extlw_6a_tau ^NEB ",
    "chi2_6a_tau ^NEB ",
    "L1400_6a_tau ^NEB (erg/s/Hz)" ,
    "L2700_6a_tau ^NEB (erg/s/Hz)" ,
    "UMag_6a_tau ^NEB (mag)" ,
    "BMag_6a_tau ^NEB (mag)" ,
    "VMag_6a_tau ^NEB (mag)" ,
    "RMag_6a_tau ^NEB (mag)" ,
    "IMag_6a_tau ^NEB (mag)" ,
    "JMag_6a_tau ^NEB (mag)" ,
    "KMag_6a_tau ^NEB (mag)" ,
    "age_6a_deltau (dex(yr))" ,
    "tau_6a_deltau (Gyr)" ,
    "EBV_6a_deltau (mag)" ,
    "SFR_6a_deltau (solMass/yr)" ,
    "met_6a_deltau (solMet)" ,
    "extlw_6a_deltau ",
    "chi2_6a_deltau ",
    "L1400_6a_deltau (erg/s/Hz)" ,
    "L2700_6a_deltau (erg/s/Hz)" ,
    "UMag_6a_deltau (mag)" ,
    "BMag_6a_deltau (mag)" ,
    "VMag_6a_deltau (mag)" ,
    "RMag_6a_deltau (mag)" ,
    "IMag_6a_deltau (mag)" ,
    "JMag_6a_deltau (mag)" ,
    "KMag_6a_deltau (mag)" ,
    "age_6a_invtau (dex(yr))" ,
    "tau_6a_invtau (Gyr)" ,
    "EBV_6a_invtau (mag)" ,
    "SFR_6a_invtau (solMass/yr)" ,
    "met_6a_invtau (solMet)" ,
    "extlw_6a_invtau ",
    "chi2_6a_invtau ",
    "L1400_6a_invtau (erg/s/Hz)" ,
    "L2700_6a_invtau (erg/s/Hz)" ,
    "UMag_6a_invtau (mag)" ,
    "BMag_6a_invtau (mag)" ,
    "VMag_6a_invtau (mag)" ,
    "RMag_6a_invtau (mag)" ,
    "IMag_6a_invtau (mag)" ,
    "JMag_6a_invtau (mag)" ,
    "KMag_6a_invtau (mag)" ,
    "age_10c ^dust (dex(yr))" ,
    "SFH_10c ^dust ",
    "tau_10c ^dust (Gyr)" ,
    "met_10c ^dust (solMet)" ,
    "age_14a_const (dex(yr))" ,
    "EBV_14a_const (mag)" ,
    "SFR_14a_const (solMass/yr)" ,
    "q_14a_const ",
    "age_14a_lin (dex(yr))" ,
    "EBV_14a_lin (mag)" ,
    "SFR_14a_lin (solMass/yr)" ,
    "q_14a_lin ",
    "age_14a_deltau (dex(yr))" ,
    "tau_14a_deltau (Gyr)" ,
    "EBV_14a_deltau (mag)" ,
    "SFR_14a_deltau (solMass/yr)" ,
    "q_14a_deltau ",
    "age_14a_tau (dex(yr))" ,
    "tau_14a_tau (Gyr)" ,
    "EBV_14a_tau (mag)" ,
    "SFR_14a_tau (solMass/yr)" ,
    "q_14a_tau "]

    

    mass = 'Data/raw/hlsp_candels_hst_wfc3_egs_multi_v1_mass-cat.txt'

    mass_data = pd.read_csv(mass, sep='\s+', header=None, skiprows=39)


    #A continuación añadimos las columnas

    mass_data.columns = ["ID"
    ,"RAdeg (deg) "
    ,"DECdeg (deg) "
    ,"Hmag (mag) "
    ,"PhotFlag "
    ,"Class_star "
    ,"AGNflag "
    ,"zphot "
    ,"zspec "
    ,"q_zspec "
    ,"r_zspec "
    ,"zbestM "
    ,"zphot_l68 "
    ,"zphot_u68 "
    ,"zphot_l95 "
    ,"zphot_u95 "
    ,"zAGN "
    ,"M_neb_med (dex(solMass)) "
    ,"s_neb_med (dex(solMass)) "
    ,"M_med (dex(solMass)) "
    ,"s_med (dex(solMass)) "
    ,"M_14a_cons (dex(solMass)) "
    ,"M_11a_tau (dex(solMass)) "
    ,"M_6a_tau^NEB (dex(solMass)) "
    ,"M_13a_tau (dex(solMass)) "
    ,"M_12a_tau (dex(solMass)) "
    ,"M_6a_tau (dex(solMass)) "
    ,"M_2a_tau (dex(solMass)) "
    ,"M_15a (dex(solMass)) "
    ,"M_10c (dex(solMass)) "
    ,"M_14a_lin (dex(solMass)) "
    ,"M_14a_deltau (dex(solMass)) "
    ,"M_14a_tau (dex(solMass)) "
    ,"M_14a_inctau (dex(solMass)) "
    ,"M_14a (dex(solMass)) "
    ,"M_neb_med_lin (solMass) "
    ,"s_neb_med_lin (solMass) "
    ,"M_med_lin (solMass) "
    ,"s_med_lin (solMass)" ]

    phot = 'Data/raw/hlsp_candels_hst_wfc3_egs_multi_v2_redshift-cat.txt'

    phot_data = pd.read_csv(phot, sep='\s+', header=None, skiprows=61)


    #A continuación añadimos las columnas

    phot_data.columns = [" file "
    ,"ID"
    ,"RA"
    ,"DEC"
    ,"z_best "
    ,"z_best_type "
    ,"z_spec "
    ,"z_spec_ref "
    ,"z_grism "
    ,"mFDa4_z_peak "
    ,"mFDa4_z_weight "
    ,"mFDa4_z683_low "
    ,"mFDa4_z683_high "
    ,"mFDa4_z954_low "
    ,"mFDa4_z954_high "
    ,"HB4_z_peak "
    ,"HB4_z_weight "
    ,"HB4_z683_low "
    ,"HB4_z683_high "
    ,"HB4_z954_low "
    ,"HB4_z954_high "
    ,"Finkelstein_z_peak "
    ,"Finkelstein_z_weight "
    ,"Finkelstein_z683_low "
    ,"Finkelstein_z683_high "
    ,"Finkelstein_z954_low "
    ,"Finkelstein_z954_high "
    ,"Fontana_z_peak "
    ,"Fontana_z_weight "
    ,"Fontana_z683_low "
    ,"Fontana_z683_high "
    ,"Fontana_z954_low "
    ,"Fontana_z954_high "
    ,"Pforr_z_peak "
    ,"Pforr_z_weight "
    ,"Pforr_z683_low "
    ,"Pforr_z683_high "
    ,"Pforr_z954_low "
    ,"Pforr_z954_high "
    ,"Salvato_z_peak "
    ,"Salvato_z_weight "
    ,"Salvato_z683_low "
    ,"Salvato_z683_high "
    ,"Salvato_z954_low "
    ,"Salvato_z954_high "
    ,"Wiklind_z_peak "
    ,"Wiklind_z_weight "
    ,"Wiklind_z683_low "
    ,"Wiklind_z683_high "
    ,"Wiklind_z954_low "
    ,"Wiklind_z954_high "
    ,"Wuyts_z_peak "
    ,"Wuyts_z_weight "
    ,"Wuyts_z683_low "
    ,"Wuyts_z683_high "
    ,"Wuyts_z954_low "
    ,"Wuyts_z954_high" ]

    def filter_redsift(z_min,z_max,phot_data,mass_data,phys_prop_data):
        """ Esta función filtra los datos para un rango de valores de z. Teniendo en cuenta que los valores de z son diferentes 
        para datasets de Photometry y Mass y que Physical Properties no tiene este dato, por lo que el filtro consiste en que esté en unos de los otros dos datasets. """
        a = phot_data[(z_min<=phot_data["z_best "])& (phot_data["z_best "]<=z_max)]
        b = mass_data[(z_min<=mass_data["zbestM "])& (mass_data["zbestM "]<=z_max)]
        c = phys_prop_data[phys_prop_data["ID"].isin(pd.concat([a["ID"], b["ID"]]).unique())]
        return a, b, c

    cosmic_noon_phot, cosmic_noon_mass, cosmic_noon_phys_prop = filter_redsift(1.5,2.5,phot_data,mass_data,phys_prop_data)

    merged1 = pd.merge(cosmic_noon_phot,cosmic_noon_mass, left_on = 'ID', right_on = 'ID', how = 'inner')

    merged2 = pd.merge(cosmic_noon_phys_prop,merged1, left_on = 'ID', right_on = 'ID', how = 'inner')

    # Ahora eliminimaos aquellas galaxias que tienen un AGN, ya que son pocas y no podemos comparar su comportamiento con el de otras que no tienen AGN.
    value_to_drop = 1
    column_name = 'AGNflag '


    df_cleaned = merged2.drop(merged2[merged2[column_name] == value_to_drop].index)

    #df_cleaned.head()
    df_cleaned.dropna(inplace=True, axis=0)
    df_cleaned['z_best'] = df_cleaned['z_best '].astype(np.float32)  # Remove space from column name
    df_cleaned.drop('z_best ', axis=1, inplace=True)  # Drop old column with space
    df_numeric =df_cleaned.select_dtypes(include=['number']).astype(np.float32)
    
    # Select mag columns using list comprehension
    mag_cols = [col for col in df_numeric.columns if '(mag)' in col]
    m_cols = [col for col in df_numeric.columns if 'M_' in col]
    age_cols = [col for col in df_numeric.columns if 'yr' in col or 'age' in col]

    # Clean values in-place using .where() for better performance
    df_numeric[mag_cols] = df_numeric[mag_cols].where(df_numeric[mag_cols] >= -50, np.nan)
    df_numeric[m_cols] = df_numeric[m_cols].where(df_numeric[m_cols] >= 0, np.nan)
    df_numeric[age_cols] = df_numeric[age_cols].where(df_numeric[age_cols] >= 0, np.nan)

    # Drop rows with any NaN values in mag columns
    df_numeric.dropna(subset=mag_cols, how='any', inplace=True)
    df_numeric.dropna(subset=m_cols, how='any', inplace=True)
    df_numeric.dropna(subset=age_cols, how='any', inplace=True)
 
    regexdic = {'Salvato': r'_11a|^z_best|^Salvato_z_peak', 'BooMeLee': r'_14a|^z_best', 'Barro': r'_2a|^z_best','Finkelstein': r'_4b|^z_best|^Finkelstein_z_peak','Pforr': r'_10c|^z_best|^Pforr_z_peak','Wikilind': r'_12a|^z_best|^Wiklind_z_peak','Wuyts': r'_13a|^z_best|^Wuyts_z_peak', 'Fontana': r'_6a|^z_best|^Fontana_z_peak'}
    df_drop = df_numeric.filter(regex=r'_low|_high|_weight|^chi2|spec|_grism|RA|DEC|ID|^zphot|Flag|^q_|^s_|AGN')
    df_numeric.drop(df_drop.columns, axis=1, inplace=True)
   
    #regexdic_test = {'Fontana': r'_6a|^z_best|^Fontana_z_peak'}
    #df_numeric_test = df_numeric.filter(regex=regexdic_test['Fontana'])
    # Fix 2: Update regex patterns to match actual column names

    # Fix 3: Verify column names before processing
    print("Columns in cleaned_numeric:", df_numeric.columns.tolist())
    print("Number of columns in cleaned_numeric:", len(df_numeric.columns.tolist()))
    print(df_numeric.shape)
    # 2D Relations (x, y)
    relations_2d = [
    ("age_6a_tau (dex(yr))", "SFR_6a_tau (solMass/yr)"),
    ("met_6a_tau (solMet)", "SFR_6a_tau (solMass/yr)"),
    ("EBV_6a_tau (mag)", "SFR_6a_tau (solMass/yr)"),
    ("Av_2a_tau (mag)", "SFR_2a_tau (solMass/yr)"),
    ("M_11a_tau (dex(solMass)) ", "SFR_11a_tau (solMass/yr)"),
    ("M_14a_tau (dex(solMass)) ", "SFR_14a_tau (solMass/yr)"),
    ("M_6a_tau (dex(solMass)) ", "met_6a_tau (solMet)"),
    ("M_6a_tau (dex(solMass)) ", "EBV_6a_tau (mag)"),
    ("M_6a_tau (dex(solMass)) ", "Lbol_12a (dex(solLum))"),
    ("M_6a_tau (dex(solMass)) ", "UMag_6a_tau (mag)"),
    ("M_6a_tau (dex(solMass)) ", "KMag_6a_tau (mag)"),
    ("M_6a_tau (dex(solMass)) ", "tau_6a_tau (Gyr)"),
    ("M_6a_tau (dex(solMass)) ", "age_6a_tau (dex(yr))"),
    ("M_6a_tau (dex(solMass)) ", "SFR_6a_tau (solMass/yr)"),
    ("M_6a_tau (dex(solMass)) ", "Av_6a_tau (mag)"),
    ("M_6a_tau (dex(solMass)) ", "L1400_6a_tau (erg/s/Hz)"),
    ("M_6a_tau (dex(solMass)) ", "L2700_6a_tau (erg/s/Hz)"),
    ("M_6a_tau (dex(solMass)) ", "UMag_6a_tau (mag)"),
    ("M_6a_tau (dex(solMass)) ", "BMag_6a_tau (mag)"),
    ("M_6a_tau (dex(solMass)) ", "VMag_6a_tau (mag)"),
    ("M_6a_tau (dex(solMass)) ", "RMag_6a_tau (mag)"),
    ("M_6a_tau (dex(solMass)) ", "IMag_6a_tau (mag)"),
    ("M_6a_tau (dex(solMass)) ", "JMag_6a_tau (mag)"),
    ("M_6a_tau (dex(solMass)) ", "KMag_6a_tau (mag)")
]

# 3D Relations (x, y, z)
    relations_3d = [
    ("age_6a_tau (dex(yr))", "SFR_6a_tau (solMass/yr)", "met_6a_tau (solMet)"),
    ("M_6a_tau (dex(solMass)) ", "SFR_6a_tau (solMass/yr)", "met_6a_tau (solMet)"),
    ("M_6a_tau (dex(solMass)) ", "age_6a_tau (dex(yr))", "met_6a_tau (solMet)"),
    ("M_6a_tau (dex(solMass)) ", "SFR_6a_tau (solMass/yr)", "EBV_6a_tau (mag)"),
    ("M_6a_tau (dex(solMass)) ", "SFR_6a_tau (solMass/yr)", "tau_6a_tau (Gyr)"),
    ("M_6a_tau (dex(solMass)) ", "SFR_6a_tau (solMass/yr)", "Lbol_12a (dex(solLum))"),
    ("M_6a_tau (dex(solMass)) ", "SFR_6a_tau (solMass/yr)", "chi2_6a_tau"),
    ("M_6a_tau (dex(solMass)) ", "SFR_6a_tau (solMass/yr)", "L1400_6a_tau (erg/s/Hz)"),
    ("M_6a_tau (dex(solMass)) ", "SFR_6a_tau (solMass/yr)", "L2700_6a_tau (erg/s/Hz)"),
    ("M_6a_tau (dex(solMass)) ", "SFR_6a_tau (solMass/yr)", "UMag_6a_tau (mag)")
]

    CHECKPOINT_FILE = f"2d_plots.txt"
    completed_plots = load_completed_plots()
    for x, y in relations_2d:

        plot2d(df_numeric, x, y, '2D',sanitize_filename(x+y), 'z_best', 'test')
    for x, y, z in relations_3d:
        plot3d(df_numeric, x, y,z, '3D',sanitize_filename(x+y+z), 'z_best', 'test')