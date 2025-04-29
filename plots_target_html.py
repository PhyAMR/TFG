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
import sympy
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.model_selection import train_test_split
from pysr import PySRRegressor
from sklearn.metrics import r2_score

class PlotCollector:
    def __init__(self):
        self.figures = []
        self.plot_count = 0
        
    def add_plot(self, fig):
        self.figures.append(fig)
        self.plot_count += 1

def create_combined_html(plot_collector, filename="all_plots.html"):
    if len(plot_collector.figures) != 0:
        html_str = '''<!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>All Interactive Plots</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                .plot-container { 
                    margin: 20px;
                    padding: 20px;
                    border: 1px solid #ddd;
                    page-break-inside: avoid;
                }
                h2.plot-title {
                    margin-bottom: 10px;
                    color: #2c3e50;
                }
            </style>
        </head>
        <body>
        <h1 style="text-align: center;">Interactive Analysis Plots</h1>
        '''
        
        for idx, fig in enumerate(plot_collector.figures):
            plot_html = fig.to_html(
                full_html=False,
                include_plotlyjs='cdn',
                div_id=f'plot-{idx}'
            )
            html_str += f'''
            <div class="plot-container">
                <h2 class="plot-title">Plot {idx+1} of {plot_collector.plot_count}</h2>
                {plot_html}
            </div>
            '''
        
        html_str += '</body></html>'
        
        with open(filename, 'w') as f:
            f.write(html_str)
        logger.info(f"Saved combined plots to {filename}")



def symbolic_regression(df, input_cols, target_col, threshold=0.8):
    X = df[input_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=12
    )

    model = PySRRegressor(
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["exp", "log", "sqrt"],
        model_selection="best",
        verbosity=0,
        constraints={'^': (-2, 2)}
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)

    if score >= threshold:
        best_expr = model.sympy()
        return best_expr, score
    else:
        return None, score

def remove_empty_directories(start_dir):
    """
    Recursively removes all empty directories starting from the given directory.
    
    Args:
        start_dir (str): The path of the directory to start cleaning from.
    """
    for root, dirs, files in os.walk(start_dir, topdown=False):
        current_dir = root
        try:
            # Check if the directory is empty
            if not os.listdir(current_dir):
                os.rmdir(current_dir)
                print(f"Removed empty directory: {current_dir}")
        except OSError as e:
            print(f"Error processing {current_dir}: {e.strerror}")
        except Exception as e:
            print(f"Unexpected error occurred while processing {current_dir}: {e}")

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
            "mag",
            "erg_s_Hz"       
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


# ... (keep all imports except remove matplotlib and add plotly)


def plot2d(df, x, y, dire, n, heatmap_column, dataset_dir,plot_collector):
    try:
        logger.info(f"Creating interactive 2D plot {n} for {x} vs {y}")
        x_positive = not np.any(df[x] <= 0)
        y_positive = not np.any(df[y] <= 0)
        
        axis_options = {
            'x': [False, True] if x_positive else [False],
            'y': [False, True] if y_positive else [False]
        }
        combinations = list(itertools.product(axis_options['x'], axis_options['y']))
        
        best_chi2 = np.inf
        best_params = None
        best_expr = None
        best_data = None
        
        for log_x, log_y in combinations:
            try:
                x_trans = np.log10(df[x]) if log_x else df[x]
                y_trans = np.log10(df[y]) if log_y else df[y]
                temp_df = pd.DataFrame({'input': x_trans, 'target': y_trans})
                
                expr, chi2 = symbolic_regression(temp_df, ['input'], 'target')
                if chi2 < best_chi2 and expr is not None:
                    best_chi2 = chi2
                    best_params = (log_x, log_y)
                    best_expr = expr
                    best_data = temp_df.copy()
            except Exception as e:
                logger.warning(f"Failed combination ({log_x}, {log_y}): {e}")
        
        if best_params is None:
            logger.info("No valid models found for any combination")
            return
            
        log_x_best, log_y_best = best_params
        safe_n = sanitize_filename(n)
        if safe_n in completed_plots:
            logger.info(f"Skipping plot {n}: already completed.")
            return

        # Create Plotly figure
        fig = go.Figure()
        # Set axis labels and titles
        x_label = f"log10({x})" if log_x_best else x
        y_label = f"log10({y})" if log_y_best else y
        fig.update_layout(
            title=f"Best Symbolic Fit: {x} vs {y}",
            xaxis_title=x_label,
            yaxis_title=y_label,
            showlegend=True,
            template='plotly_white',
            annotations=[
                dict(
                    x=0.05,
                    y=0.95,
                    xref='paper',
                    yref='paper',
                    text=f"{best_expr}<br>RMSE: {best_chi2:.2f}",
                    showarrow=False,
                    bgcolor='white',
                    bordercolor='black',
                    borderwidth=1
                )
            ]
        
        )
        # In plot2d function:
        if heatmap_column and heatmap_column in df:
            # Use .loc to ensure alignment with original indices
            colors = df.loc[best_data.index, heatmap_column].values
            fig.add_trace(go.Scatter(
                x=best_data['input'].values,
                y=best_data['target'].values,
                mode='markers',
                marker=dict(
                    color=colors,  # Now properly aligned
                    colorscale='Viridis',
                    showscale=True,
                    size=10,  # Increased size
                    opacity=0.8
                ),
                name='Data'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=best_data['input'].values,
                y=best_data['target'].values,
                mode='markers',
                marker=dict(
                    color='Blue',  # More visible color
                    size=2,
                    opacity=0.8
                ),
                name='Data'
            ))
        
        # Add regression line
        try:
            x_vals = np.linspace(best_data['input'].min(), best_data['input'].max(), 100)
            expr_func = sympy.lambdify(sympy.Symbol('x0'), best_expr, 'numpy')
            y_vals = expr_func(x_vals)
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                line=dict(color='red', width=3),
                name='Best Fit'
            ))
        except Exception as e:
            logger.warning(f"Could not plot regression line: {e}")
        
        
        
        
        
        plot_collector.add_plot(fig)
        save_completed_plot(safe_n)
        return fig
        
    except Exception as e:
        logger.error(f"Error in optimized 2D plot: {e}", exc_info=True)
        return None

def plot3d(df, x, y, z, dire, n, heatmap_column, dataset_dir,plot_collector):
    try:
        logger.info(f"Creating interactive 3D plot {n} for {x}, {y}, {z}")
        safe_n = sanitize_filename(n)
        if safe_n in completed_plots:
            logger.info(f"Skipping plot {n}: already completed.")
            return
        
        eligible = [np.all(df[col] > 0) for col in [x, y, z]]
        axis_options = [[False, True] if e else [False] for e in eligible]
        combinations = list(itertools.product(*axis_options))
        
        best_chi2 = np.inf
        best_params = None
        best_expr = None
        best_data = None
        
        for log_x, log_y, log_z in combinations:
            try:
                x_trans = np.log10(df[x]) if log_x else df[x]
                y_trans = np.log10(df[y]) if log_y else df[y]
                z_trans = np.log10(df[z]) if log_z else df[z]
                temp_df = pd.DataFrame({
                    'x': x_trans, 
                    'y': y_trans, 
                    'z': z_trans
                })
                
                expr, chi2 = symbolic_regression(temp_df, ['x', 'y'], 'z')
                if chi2 < best_chi2 and expr is not None:
                    best_chi2 = chi2
                    best_params = (log_x, log_y, log_z)
                    best_expr = expr
                    best_data = temp_df.copy()
            except Exception as e:
                logger.warning(f"Failed combination {log_x, log_y, log_z}: {e}")
        
        if best_params is None:
            logger.info("No valid 3D models found")
            return
            
        log_x_best, log_y_best, log_z_best = best_params
        
        # Create Plotly figure
        fig = go.Figure()
        # Set axis labels and titles
        x_label = f"log10({x})" if log_x_best else x
        y_label = f"log10({y})" if log_y_best else y
        z_label = f"log10({z})" if log_z_best else z
        
        fig.update_layout(
            title=f"Best 3D Symbolic Fit: {z} = f({x}, {y})",
            scene=dict(
                xaxis_title=x_label,
                yaxis_title=y_label,
                zaxis_title=z_label
            ),
            annotations=[
                    dict(
                        x=0.05,
                        y=0.95,
                        text=f"{best_expr}<br>RMSE: {best_chi2:.2f}",
                        showarrow=False,
                        font=dict(size=12),
                        bordercolor='black',
                        borderwidth=1,
                        bgcolor='white'
                    )
                ],
            margin=dict(l=0, r=0, b=0, t=40),
            template='plotly_white'
        )
        # In plot3d function:
        if heatmap_column and heatmap_column in df:
            colors = df.loc[best_data.index, heatmap_column].values
            fig.add_trace(go.Scatter3d(
                x=best_data['x'].values,
                y=best_data['y'].values,
                z=best_data['z'].values,
                mode='markers',
                marker=dict(
                    color=colors,
                    colorscale='Viridis',
                    size=2,
                    opacity=0.8,
                    showscale=True
                ),
                name='Data'
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=best_data['x'].values,
                y=best_data['y'].values,
                z=best_data['z'].values,
                mode='markers',
                marker=dict(
                    color='Blue',
                    size=2,
                    opacity=0.8
                ),
                name='Data'
            ))
        
        # Add regression surface
        try:
            x_sym, y_sym = sympy.symbols('x0 x1')
            expr_func = sympy.lambdify((x_sym, y_sym), best_expr, 'numpy')
            x_vals = np.linspace(best_data['x'].min(), best_data['x'].max(), 20)
            y_vals = np.linspace(best_data['y'].min(), best_data['y'].max(), 20)
            X, Y = np.meshgrid(x_vals, y_vals)
            Z = expr_func(X, Y)
            
            fig.add_trace(go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale='Reds',
                opacity=0.5,
                showscale=False,
                name='Best Fit'
            ))
        except Exception as e:
            logger.warning(f"Could not plot 3D surface: {e}")
        
        
        
        plot_collector.add_plot(fig)
        save_completed_plot(safe_n)
        return fig
        
    except Exception as e:
        logger.error(f"Error in optimized 3D plot: {e}", exc_info=True)
        return None

def process_dataframe(df: pd.DataFrame,
                     column_regex_config: dict,
                     base_output_dir: str = "D:/analysis",
                     heatmap_column: str = "Heat",
                     include_base: bool = True,
                     target: str = "Objective"
                     ) -> None:
    """
    Process a DataFrame by creating column subsets using column name regex patterns
    
    Args:
        df: Input DataFrame to process
        column_regex_config: Dictionary of {subset_name: regex_pattern} for column selection
        base_output_dir: Root directory for output
        heatmap_column: Column to use for heatmap coloring (automatically included)
        include_base: Whether to process original DataFrame without filtering
    """

    
    try:
        logger.info(f"Starting processing of DataFrame with {len(df.columns)} columns")
        
        # Create list of datasets to process
        datasets = []
        
        # Create column subsets
        for subset_name, col_regex in column_regex_config.items():
            try:
                safe_subset = sanitize_filename(subset_name)
                subset_cols = df.filter(regex=col_regex).columns.tolist()
                
                if (heatmap_column and heatmap_column not in subset_cols) or (target and target not in subset_cols):
                    print("")
                    #subset_cols.append(heatmap_column)
                    #subset_cols.append(target)
                    continue
                
                if not subset_cols:
                    logger.warning(f"Skipping {subset_name} - no columns matched regex")
                    continue
                
                subset_df = df[subset_cols].dropna(how='all', axis=1)
                
                if subset_df.empty:
                    logger.warning(f"Skipping {subset_name} - resulting DataFrame is empty")
                    continue
                
                
                datasets.append((safe_subset, subset_df))
                logger.info(f"Created subset {safe_subset} with columns: {subset_cols}")

            except Exception as e:
                logger.error(f"Error creating subset {subset_name}: {str(e)}", exc_info=True)
                continue

        # Add base dataset if requested
        if include_base:
            datasets.append(("base_dataset", df))
            logger.info("Including base dataset with all columns")

        # Track plotted combinations
        plotted_2d = set()
        plotted_3d = set()

        # Process all datasets
        logger.info(f"Processing {len(datasets)} column subsets")
        for dataset_name, dataset_df in datasets:
            plot_collector = PlotCollector()
            
            try:
                if target and target not in subset_df.columns:
                    logger.warning(f"Skipping {subset_name} - target column not found")
                    continue
                dataset_dir = os.path.normpath(os.path.join(base_output_dir, dataset_name))
                subdirs = ['pairplots', '2d_plots', '3d_plots']
                
                for subdir in subdirs:
                    dir_path = os.path.join(dataset_dir, subdir)
                    os.makedirs(dir_path, exist_ok=True)

                logger.info(f"Processing {dataset_name} (columns: {len(dataset_df.columns)})")
                
                

                numeric_cols = dataset_df.select_dtypes(include=np.number).columns.tolist()
                if heatmap_column in numeric_cols and not np.issubdtype(dataset_df[heatmap_column].dtype, np.number):
                    numeric_cols.remove(heatmap_column)

                # 2D plots with tracking
                if len(numeric_cols) >= 2:
                    plot2d_dir = os.path.join(dataset_dir, "2d_plots")
                    for i in numeric_cols:
                        combo_key = (i, target)
                        if combo_key not in plotted_2d:
                            safe_x = sanitize_filename(i)
                            safe_y = sanitize_filename(target)
                            plot_name = f"{dataset_name}_{safe_x}_vs_{safe_y}"
                            if plot_condition(plot_name):
                                plot2d(dataset_df, i, target, plot2d_dir, plot_name, heatmap_column,dataset_dir,plot_collector)
                            else:
                                logger.info(f"Skipping 2D plot {plot_name} - columns not useful")
                            plotted_2d.add(combo_key)
                            plotted_2d.add((target, i))  # Add reverse to prevent mirror plots
                        else:
                            logger.info(f"Skipping 2D plot {plot_name} - already plotted")

                # 3D plots with tracking
                if len(numeric_cols) >= 3:
                    plot3d_dir = os.path.join(dataset_dir, "3d_plots")
                    for combo in itertools.combinations(numeric_cols, 2):
                        combo_key = tuple(sorted(combo))  # Sort for consistent tracking
                        if combo_key not in plotted_3d:
                            safe_combo = [sanitize_filename(col) for col in combo]
                            plot_name = f"{dataset_name}_{'_'.join(safe_combo)}"
                            if plot_condition(plot_name):
                                plot3d(dataset_df, *combo,target, plot3d_dir, plot_name, heatmap_column, dataset_dir,plot_collector)
                            else:
                                logger.info(f"Skipping 3D plot {plot_name} - columns not useful")
                            plotted_3d.add(combo_key)
                        else:
                            logger.info(f"Skipping 3D plot {plot_name} - already plotted")

            except Exception as e:
                logger.error(f"Error processing {dataset_name}: {str(e)}", exc_info=True)
                continue
            create_combined_html(plot_collector, f"{dataset_dir}/{sanitize_filename(target)}_{dataset_name}_plots.html")
        #logger.info(f"______________Starting pairplots_______________")
        #for dataset_name, dataset_df in datasets:
        #    try:
        #        dataset_dir = os.path.normpath(os.path.join(base_output_dir, dataset_name))
        #        subdirs = ['pairplots', '2d_plots', '3d_plots']
        #        
        #        for subdir in subdirs:
        #            dir_path = os.path.join(dataset_dir, subdir)
        #            os.makedirs(dir_path, exist_ok=True)
#
        #        logger.info(f"Processing {dataset_name} (columns: {len(dataset_df.columns)})")
        #        
        #        # Pairplot
        #        if len(dataset_df.select_dtypes(include=np.number).columns) >= 2:
        #            custom_pairplot(dataset_df, heatmap_column, 
        #                           os.path.join(dataset_dir, "pairplots"), 
        #                           dataset_name)
        #    except Exception as e:
        #        logger.error(f"Error processing {dataset_name}: {str(e)}", exc_info=True)
        #        continue
#
        #logger.info("Column subset processing completed")

    except Exception as e:
        logger.error(f"Critical error in process_dataframe: {str(e)}", exc_info=True)
        raise
    

def custom_pairplot(df, heatmap_column, output_dir, title):
    """Enhanced pairplot with robust KDE handling."""
    try:
        logger.info(f"Creating pairplot for {title}")
        
        numeric_df = df.select_dtypes(include=np.number)
        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(df[heatmap_column].min(), df[heatmap_column].max())

        # Create PairGrid with proper color mapping
        g = sns.PairGrid(numeric_df, diag_sharey=False)
        
        # Upper triangle: Scatter plots with heatmap coloring
        def scatter_heatmap(x, y, **kws):
            kws.pop('color', None)
            plt.scatter(
                x, y,
                c=df[heatmap_column],
                cmap=cmap,
                norm=norm,
                alpha=0.7,
                **kws
            )
        g.map_upper(scatter_heatmap)

        # Lower triangle: Density plots with error handling
        def safe_kdeplot(x, y, **kws):
            """KDE plot with enforced level ordering"""
            try:
                # Calculate reasonable levels
                levels = np.linspace(0, 1, 5)
                sns.kdeplot(
                    x=x, y=y,
                    cmap="Blues",
                    fill=True,
                    levels=levels,
                    thresh=0.05,
                    **kws
                )
            except Exception as kde_error:
                logger.warning(f"KDE plot failed: {str(kde_error)}")
                # Fallback to scatter plot
                plt.scatter(x, y, alpha=0.3)
                
        g.map_lower(safe_kdeplot)

        # Diagonal: Histograms with KDE
        g.map_diag(sns.histplot, kde=True, color="green", alpha=0.5)

        # Add colorbar
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        g.figure.colorbar(sm, ax=g.axes.ravel().tolist(), label=heatmap_column)

        # Save plot
        filename = os.path.join(output_dir, f"{title}_pairplot.png").replace(" ", "_")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        
        logger.info(f"Saved pairplot to {filename}")

    except Exception as e:
        logger.error(f"Pairplot error: {str(e)}", exc_info=True)
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
    df_drop = df_numeric.filter(regex=r'_low|_high|_weight|^chi2|spec|_grism|RA|DEC|ID|^zphot|Flag|^q_|^s_|AGN|^EBV')
    df_numeric.drop(df_drop.columns, axis=1, inplace=True)
   
    #regexdic_test = {'Fontana': r'_6a|^z_best|^Fontana_z_peak'}
    #df_numeric_test = df_numeric.filter(regex=regexdic_test['Fontana'])
    # Fix 2: Update regex patterns to match actual column names

    # Fix 3: Verify column names before processing
    print("Columns in cleaned_numeric:", df_numeric.columns.tolist())
    print("Number of columns in cleaned_numeric:", len(df_numeric.columns.tolist()))
    

    targets = df_numeric.filter(regex=r'^M_\d+|^SFR|^L').columns.tolist()
        
    targets_df = df_numeric.filter(regex=r'^M_\d+|^SFR|^L|(mag)|met|z_best')
    print(targets)
    print(len(targets))
    for t in targets:
        tar = sanitize_filename(t)

        # Fix 4: Update process_dataframe call

        CHECKPOINT_FILE = f"{tar}_html_plots.txt"

        

        completed_plots = load_completed_plots()
        process_dataframe(
        targets_df, 
        regexdic,
        f"D:/TFG/plots_html/{tar}", 
        heatmap_column='z_best',  # Use corrected column name
        include_base=False,
        target=t

    )
    remove_empty_directories(f"D:/TFG/plots_html/{tar}") 