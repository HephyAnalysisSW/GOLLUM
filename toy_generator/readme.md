
### **Interface: `run_pseudo_experiments`**

This interface is designed to streamline the process of generating pseudo experiments with minimal user input. Below is the detailed explanation of its parameters and usage.

### **Function Signature**

def run_pseudo_experiments(
    tes,
    jes,
    soft_met,
    ttbar_scale,
    diboson_scale,
    bkg_scale,
    num_pseudo_experiments,
    num_of_sets,
    ground_truth_mus,
    output_dir,
)


### **Parameters**

1. **`tes` (float)**:
   - **Description**: Tau Energy Scale factor.
   - **Purpose**: Adjusts the energy scale for taus in the dataset. If set to `None`, tes is set to 1.
   - **Example**: 
     - `tes=1.05` increases the tau energy scale by 5%.
     - `tes=None` tes is set to 1.

2. **`jes` (float)**:
   - **Description**: Jet Energy Scale factor.
   - **Purpose**: Adjusts the energy scale for jets. If set to `None`, jes is set to 1.
   - **Example**: 
     - `jes=0.97` reduces the jet energy scale by 3%.
     - `jes=None` jes is set to 1.

3. **`soft_met` (float)**:
   - **Description**: additional noise source in measuring Soft MET.
   - **Purpose**: Adds or adjusts the soft MET in the dataset. If set to `None`, soft_met is set to 1.
   - **Example**: 
     - `soft_met=0.8` applies a soft MET systematic factor of 0.8.
     - `soft_met=None` soft_met is set to 1.

4. **`ttbar_scale` (float)**:
   - **Description**: Scaling factor for the `ttbar` background process.
   - **Purpose**: Scales the contribution of `ttbar` background. If set to `None`, no scaling is applied.
   - **Example**: 
     - `ttbar_scale=1.1` increases the `ttbar` contribution by 10%.
     - `ttbar_scale=None` disables scaling for `ttbar`.

5. **`diboson_scale` (float)**:
   - **Description**: Scaling factor for the `diboson` background process.
   - **Purpose**: Scales the contribution of `diboson` background. If set to `None`, no scaling is applied.
   - **Example**: 
     - `diboson_scale=1.15` increases the `diboson` contribution by 15%.
     - `diboson_scale=None` disables scaling for `diboson`.

6. **`bkg_scale` (float)**:
   - **Description**: Scaling factor for other background processes.
   - **Purpose**: Scales the contribution of other background processes. If set to `None`, no scaling is applied.
   - **Example**: 
     - `bkg_scale=1.1` increases the overall bkg by 10%.
     - `bkg_scale=None` disables scaling for other backgrounds.

7. **`num_pseudo_experiments` (int)**:
   - **Description**: Number of pseudo experiments to generate per set.
   - **Purpose**: Controls the number of pseudo experiments for each set.
   - **Example**: 
     - `num_pseudo_experiments=5` generates 5 pseudo experiments per set.

8. **`num_of_sets` (int)**:
   - **Description**: Number of sets to generate.
   - **Purpose**: Controls the total number of sets.
   - **Example**: 
     - `num_of_sets=3` generates 3 sets of pseudo experiments.

9. **`ground_truth_mus` (list of floats)**:
   - **Description**: Ground truth mu values for each set.
   - **Purpose**: Specifies the expected signal strength (`mu`) for each set. The length of this list must match `num_of_sets`.
   - **Example**: 
     - `ground_truth_mus=[1.0, 1.2]` sets the ground truth mu values to 1.0 and 1.2 for two sets.

10. **`output_dir` (str)**:
    - **Description**: Directory to save the output HDF5 and PKL files.
    - **Purpose**: Specifies where the generated files will be saved.
    - **Example**: 
      - `output_dir="/path/to/output"` saves all generated files to the specified directory.

### **Output**
For each pseudo experiment, the interface generates:
HDF5 File:
Contains the dataset for the pseudo experiment.
File name format: set_<mu_value>_pseudo_exp_<experiment_index>.h5.
PKL File:
Contains the 6 systematics factors (tes, jes, etc.) and ground truth mu for the pseudo experiment.
File name format: set_<mu_value>_pseudo_exp_<experiment_index>.pkl.




