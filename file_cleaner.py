import os
import shutil

def organize_plots(root_dir, unit_groups):
    """
    Organizes PNG files into unit-group subdirectories within their current directory.
    
    Args:
        root_dir (str): Root directory to process
        unit_groups (dict): Dictionary mapping folder names to lists of unit patterns
    """
    print(f"Organizing plots in {root_dir}")
    r = os.walk(root_dir)
    for foldername, subdirs, filenames in r:
        # Skip directories we've created to avoid reprocessing
        subdirs[:] = [d for d in subdirs if d not in unit_groups.keys()]
        print(f"Processing {foldername}")
        print(f"Subdirectories: {subdirs}")

        png_files = [f for f in filenames if f.lower().endswith('.png')]

        for png in png_files:
            src_path = os.path.join(foldername, png).replace("\\", "/")
            moved = False
            print(f"Processing {src_path}")
            

            for group_name, patterns in unit_groups.items():
                # Count how many patterns match components in the split filename
                distinct_matches = 0
                for pattern in patterns:
                    distinct_matches += png.count(pattern)
                
                print(f"Checking patterns in group '{group_name}'")
                print(f"Number of distinct matches: {distinct_matches}")
                if "chi2" in png.lower() or "agn" in png.lower():  # Special case for chi plots
                    dest_dir = os.path.join(foldername, "chi")
                    os.makedirs(dest_dir, exist_ok=True)
                    
                    shutil.move(src_path, os.path.join(dest_dir, png))
                    print(f"Moved {png} to {dest_dir}")
                    moved = True
                    break  # Move to first matching group only

                elif distinct_matches >= 2:  # At least 2 distinct patterns must match
                    dest_dir = os.path.join(foldername, group_name)
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.move(src_path, os.path.join(dest_dir, png))
                    print(f"Moved {png} to {dest_dir}")
                    moved = True
                    break  # Move to first matching group only

def plot_condition(name, unit_groups):
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
        
        print(f"Checking patterns in group '{group_name}'")
        print(f"Number of same matches: {same_matches}")
        
        if same_matches >= 2:  # At least 2 distinct patterns must match
            return False
        
    return True


# Example Configuration
if __name__ == "__main__":
    unit_groups = {
        "Time_Plots": [
            "dex(yr)",  
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
            "dex(solLum)"  
        ],
        "Mass_Plots": [
            "dex(solMass)"  
        ]
    }

    root_directory = "D:/TFG/plots/"
    organize_plots(root_directory, unit_groups)
