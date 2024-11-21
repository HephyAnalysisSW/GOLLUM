import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import sys
import gc
import pysqlite3
sys.modules['sqlite3'] = pysqlite3
sys.modules['_sqlite3'] = pysqlite3
import seaborn as sns  # seaborn for nice plot quicker
from sklearn.metrics import roc_curve
from IPython.display import display
from sklearn.metrics import roc_auc_score
from tabulate import tabulate

class Dataset_visualise:
    """
    A class for visualizing datasets.

    Parameters:
        * data_set (dict): The dataset containing the data, labels, weights, and detailed labels.
        * name (str): The name of the dataset (default: "dataset").
        * columns (list): The list of column names to consider (default: None, which includes all columns).

    Attributes:
        * dfall (DataFrame): The dataset.
        * target (Series): The labels.
        * weights (Series): The weights.
        * detailed_label (ndarray): The detailed labels.
        * columns (list): The list of column names.
        * name (str): The name of the dataset.
        * keys (ndarray): The unique detailed labels.
        * weight_keys (dict): The weights for each detailed label.

    Methods:
        * examine_dataset(): Prints information about the dataset.
        * histogram_dataset(columns=None): Plots histograms of the dataset features.
        * correlation_plots(columns=None): Plots correlation matrices of the dataset features.
        * pair_plots(sample_size=10, columns=None): Plots pair plots of the dataset features.
        * stacked_histogram(field_name, mu_hat=1.0, bins=30): Plots a stacked histogram of a specific field in the dataset.
        * pair_plots_syst(df_syst, sample_size=10): Plots pair plots between the dataset and a system dataset.
    """

    def __init__(self, data_set, name="dataset", columns=None):
        print("\nGeneral Structure of the data object is a dictionary")

        columns_list = ["PRI_had_pt", "PRI_had_eta", "PRI_had_phi", "PRI_lep_pt", "PRI_lep_eta","PRI_lep_phi", "PRI_met", "PRI_met_phi", "PRI_jet_num", "PRI_jet_leading_pt","PRI_jet_leading_eta", "PRI_jet_leading_phi", "PRI_jet_subleading_pt","PRI_jet_subleading_eta", "PRI_jet_subleading_phi", "PRI_jet_all_pt","DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h","DER_deltaeta_jet_jet", "DER_mass_jet_jet", "DER_prodeta_jet_jet","DER_deltar_had_lep", "DER_pt_tot", "DER_sum_pt", "DER_pt_ratio_lep_tau","DER_met_phi_centrality", "DER_lep_eta_centrality"]
#        custom_pretty_print(data_set)
        if isinstance(data_set, str):
            with h5py.File(data_set, "r") as hf:
                self.dfall = pd.DataFrame(hf['data'][:], columns=columns_list)
                self.target = pd.Series(hf['labels'][:])
                self.weights = pd.Series(hf['weights'][:])
                self.detailed_label = pd.Series(hf['detailed_labels'][:])
        elif isinstance(data_set, dict):
            self.dfall = data_set["data"]
            self.target = data_set["labels"]
            self.weights = data_set["weights"]
            self.detailed_label = pd.Series(data_set["detailed_labels"])
            self.dfall.columns = columns_list
            print(f"[DEBUG] Columns in dfall after loading from dictionary: {self.dfall.columns.tolist()}")
        if columns == None:
            self.columns = self.dfall.columns
        else:
            self.columns = columns
        self.name = name
        self.keys = np.unique(self.detailed_label)
        self.weight_keys = {}
        for key in self.keys:
            self.weight_keys[key] = self.weights[self.detailed_label == key]

    def examine_dataset(self):
        """
        Prints information about the dataset.
        """

        print()
        
        info_dict = {
            "Dataset name": self.name,
            "Number of events": self.dfall.shape[0],
            "Number of features": self.dfall.shape[1],
        }
        
        print(tabulate(info_dict.items(), headers=["Key", "Value"], tablefmt='grid'),"\n")

        weight_dict = {}
        for key in self.keys:
            weight_dict[key] = (np.sum(self.weight_keys[key]),len(self.weight_keys[key]))
            
        table_data = []
        for key in self.keys:
            table_data.append([key, weight_dict[key][0], weight_dict[key][1]])
        
        table_data.append(["Total Signal", np.sum(self.weights[self.target == 1]), len(self.weights[self.target == 1])])
        table_data.append(["Total Background", np.sum(self.weights[self.target == 0]), len(self.weights[self.target == 0])])
            
        print("[*] --- Detailed Label Summary")

        print(tabulate(table_data, headers=["Detailed Label", "Total Weight", "Number of events"], tablefmt='grid'))

        print("\n[*] --- Examples of all features\n")
        display(self.dfall.head())

        print("\n[*] --- Description of all features\n")
        display(self.dfall.describe())

    def histogram_dataset(self, columns=None,nbin = 25):
        """
        Plots histograms of the dataset features.

        Args:
            * columns (list): The list of column names to consider (default: None, which includes all columns).
            * nbin (int): The number of bins for the histogram (default: 25).
            
        .. Image:: ../images/histogram_datasets.png
        """
        
        if columns is None:
            columns = self.columns
        sns.set_theme(style="whitegrid")

        df = pd.DataFrame(self.dfall, columns=columns)
        
        
        # Number of rows and columns in the subplot grid
        n_cols = 2  # Number of columns in the subplot grid
        n_rows = int(np.ceil(len(columns) / n_cols))  # Calculate the number of rows needed

        # Create a figure and a grid of subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(17, 6 * n_rows))
        axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy indexing

        for i, column in enumerate(columns):
            # Determine the combined range for the current column

            lower_percentile = 0
            upper_percentile = 97.5

            data_array = self.dfall[column].values 
            lower_bound = np.percentile(data_array, lower_percentile)
            upper_bound = np.percentile(data_array, upper_percentile)
            mask = (data_array >= lower_bound) & (data_array <= upper_bound)
            data_clipped = data_array[mask]
            weights_clipped = self.weights.values[mask]
            target_clipped = self.target.values[mask]
            min_value = data_clipped.min()
            max_value = data_clipped.max()
        #    lower_bound = np.percentile(df[column], lower_percentile)
        #    upper_bound = np.percentile(df[column], upper_percentile)
         
        #    df_clipped = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        #    weights_clipped = self.weights[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        #    target_clipped = self.target[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            
        #    min_value = df_clipped[column].min()
        #    max_value = df_clipped[column].max()

            # Define the bin edges
            bin_edges = np.linspace(min_value, max_value, nbin + 1)
            
            signal_field = data_clipped[target_clipped == 1]
            background_field = data_clipped[target_clipped == 0]
            signal_weights = weights_clipped[target_clipped == 1]
            background_weights = weights_clipped[target_clipped == 0]
            
            # Plot the histogram for label == 1 (Signal)
            axes[i].hist(signal_field, bins=bin_edges, alpha=0.4, color='blue', label='Signal', weights=signal_weights, density=True)
            
            axes[i].hist(background_field, bins=bin_edges, alpha=0.4, color='red', label='Background', weights=background_weights, density=True)    

            
            # Set titles and labels
            axes[i].set_title(f'{column}', fontsize=16)
            axes[i].set_xlabel(column)
            axes[i].set_ylabel('Density')
            
            # Add a legend to each subplot
            axes[i].legend()

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.savefig("plot1.png")

    def correlation_plots(self, columns=None):
        """
        Plots correlation matrices of the dataset features.

        Args:
        * columns (list): The list of column names to consider (default: None, which includes all columns).

        .. Image:: ../images/correlation_plots.png
        """
        caption = ["Signal feature", "Background feature"]
        if columns is None:
            columns = self.columns
        sns.set_theme(rc={"figure.figsize": (12, 10)}, style="whitegrid")

        for i in range(2):
            mask = (self.target == i)
            dfplot = pd.DataFrame(self.dfall[mask], columns=columns)

            print(caption[i], " correlation matrix")
            corrMatrix = dfplot.corr()
            plt.figure(figsize=(12, 10))
            sns.heatmap(corrMatrix, annot=True, fmt='.2f', cmap='viridis', linewidths=0.5, annot_kws={"size": 10})
            plt.title("Correlation matrix of features in " + self.name + " - " + caption[i])
            filename = f"{self.name}_correlation_matrix_{caption[i].replace(' ', '_').lower()}.png"
            plt.savefig(filename)
            print(f"Saved figure as {filename}")
            plt.show()
            plt.close()

    def pair_plots(self, sample_size=10, columns=None):
        """
        Plots pair plots of the dataset features.

        Args:
            * sample_size (int): The number of samples to consider (default: 10).
            * columns (list): The list of column names to consider (default: None, which includes all columns).

        .. Image:: ../images/pair_plot.png
        """
        if columns is None:
            columns = self.columns
        df_sample = self.dfall[columns].copy()
        df_sample["Label"] = self.target

        df_sample_signal = df_sample[df_sample["Label"] == 1].sample(n=sample_size, random_state=42)
        df_sample_background = df_sample[df_sample["Label"] == 0].sample(n=sample_size, random_state=42)
        df_sample = pd.concat([df_sample_signal, df_sample_background])

        sns.set_theme(rc={"figure.figsize": (16, 14)}, style="whitegrid")

        ax = sns.PairGrid(df_sample, hue="Label")
        ax.map_upper(sns.scatterplot, alpha=0.5, size=0.3)
        ax.map_lower(
            sns.kdeplot, fill=True, levels=5, alpha=0.5
        )  # Change alpha value here
        ax.map_diag(
            sns.histplot,
            alpha=0.5,
            bins=20,
        )  # Change alpha value here
        ax.add_legend(title="Legend", labels=["Signal", "Background"], fontsize=12)

        legend = ax.legend
        for line in legend.get_lines():  # For lines
            line.set_alpha(0.5)
            line.set_linewidth(1.5)

        plt.rcParams["figure.facecolor"] = "w"  # Set the figure facecolor to white
        ax.figure.suptitle("Pair plots of features in " + self.name)
        plt.savefig(f"{self.name}_pair_plots.png")
        plt.show()
        plt.close()

    def stacked_histogram(self, field_name, mu_hat=1.0, bins=30,y_scale='linear'):
        """
        Plots a stacked histogram of a specific field in the dataset.

        Args:
            * field_name (str): The name of the field to plot.
            * mu_hat (float): The value of mu (default: 1.0).
            * bins (int): The number of bins for the histogram (default: 30).

        .. Image:: ../images/stacked_histogram.png
        """
        field = self.dfall[field_name]
        sns.set_theme(rc={"figure.figsize": (8, 7)}, style="whitegrid")
        lower_bound = np.percentile(field, 0)
        upper_bound = np.percentile(field, 97.5)
        mask = (field >= lower_bound) & (field <= upper_bound)
        field_clipped = field[mask]
        target_clipped = self.target[mask]
        weights_clipped = self.weights[mask]

        bins = 50

        hist_s, bins = np.histogram(
            field_clipped[target_clipped == 1],
            bins=bins,
            weights=weights_clipped[target_clipped == 1],
        )

        hist_b, bins = np.histogram(
            field_clipped[target_clipped == 0],
            bins=bins,
            weights=weights_clipped[target_clipped == 0],
        )

        hist_bkg = hist_b.copy()

        higgs = "htautau"

        for key in self.keys:
            if key != higgs:
                mask_key = (self.detailed_label == key)
                field_key = field[mask_key]
                weights_key = self.weights[mask_key]
                hist, bins = np.histogram(
                    field_key,
                    bins=bins,
                    weights=weights_key,
                )
                plt.stairs(hist_b, bins, fill=True, label=f"{key} bkg")
                hist_b -= hist
            else:
                print(key, hist_s.shape)

        plt.stairs(
            hist_s * mu_hat + hist_bkg,
            bins,
            fill=False,
            color="orange",
            label = f"$H \\rightarrow \\tau \\tau (\\mu = {mu_hat:.3f})$"
        )

        plt.stairs(
            hist_s + hist_bkg,
            bins,
            fill=False,
            color="red",
            label=f"$H \\rightarrow \\tau \\tau (\\mu = {1.0:.3f})$",
        )

        plt.legend()
        plt.title(f"Stacked histogram of {field_name} in {self.name}")
        plt.xlabel(f"{field_name}")
        plt.ylabel("Weighted count")
        plt.yscale(y_scale)
        plt.savefig(f"{self.name}_stacked_histogram_{field_name}.png")
        plt.show()

    def pair_plots_syst(self, df_syst, columns=None, sample_size=100):
        """
        Plots pair plots between the dataset and a system dataset.

        Args:
            * df_syst (DataFrame): The system dataset.
            * sample_size (int): The number of samples to consider (default: 10).
        
        ..images:: ../images/pair_plot_syst.png
        """
        if columns is None:
            columns = self.columns
        df_sample = self.dfall[columns].copy().reset_index(drop=True)
        df_sample_syst = df_syst[columns].copy().reset_index(drop=True)
        max_sample_size = min(len(df_sample), len(df_sample_syst))
        sample_size = min(sample_size, max_sample_size)
        index = np.random.choice(max_sample_size, sample_size, replace=False)
        df_sample = df_sample.iloc[index]
        df_sample_syst = df_sample_syst.iloc[index]
        df_sample["syst"] = False
        df_sample_syst["syst"] = True

        df_sample = pd.concat([df_sample, df_sample_syst])

        sns.set_theme(rc={"figure.figsize": (16, 14)}, style="whitegrid")

        ax = sns.PairGrid(df_sample, hue="syst")
        ax.map_upper(sns.scatterplot, alpha=0.5, size=0.3)
        ax.map_lower(
            sns.kdeplot, fill=True, levels=5, alpha=0.5
        )  # Change alpha value here
        ax.map_diag(
            sns.histplot,
            alpha=0.5,
            bins=20,
        )  # Change alpha value here
        ax.add_legend(title="Legend", labels=["no_syst", "syst"], fontsize=12)

        ax.figure.suptitle("Pair plots of features between syst and no_syst")
        plt.savefig(f"{self.name}_pair_plots_syst.png")
        plt.show()
        plt.close()

    def histogram_syst(self, df_nominal, weight_nominal, df_syst, weight_syst, columns=None,nbin = 25):

 #       df_sample = self.dfall[self.columns].copy()
 #       df_sample_syst = df_syst[self.columns].copy()

        
        if columns is None:
            columns = self.columns
        sns.set_theme(style="whitegrid")
        
        # Number of rows and columns in the subplot grid
        n_cols = 3
        n_rows = int(np.ceil(len(columns) / n_cols))

        # Create a figure and a grid of subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
        axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy indexing

        for i, column in enumerate(columns):

            lower_percentile = 0
            upper_percentile = 97.5
            
            lower_bound = np.percentile(df_nominal[column], lower_percentile)
            upper_bound = np.percentile(df_nominal[column], upper_percentile)

            df_clipped_nominal = df_nominal[(df_nominal[column] >= lower_bound) & (df_nominal[column] <= upper_bound)]
            weights_clipped_nominal = weight_nominal[(df_nominal[column] >= lower_bound) & (df_nominal[column] <= upper_bound)]
            
            df_clipped_syst = df_syst[(df_syst[column] >= lower_bound) & (df_syst[column] <= upper_bound)] 
            weights_clipped_syst = weight_syst[(df_syst[column] >= lower_bound) & (df_syst[column] <= upper_bound)]
            del lower_bound, upper_bound
            gc.collect()
            min_value = df_clipped_nominal[column].min()
            max_value = df_clipped_nominal[column].max()

            # Define the bin edges
            bin_edges = np.linspace(min_value, max_value, nbin + 1)
            
     #       norminal_field = df_clipped[column]
     #       syst_field = df_clipped_syst[column]

            
            # Plot the histogram for label == 1 (Signal)
            axes[i].hist(df_clipped_nominal[column], bins=bin_edges, alpha=0.6, color='blue', label='Nominal', weights=weights_clipped_nominal, density=True)
            
            axes[i].hist(df_clipped_syst[column], bins=bin_edges, alpha=0.6, color='red', label='Systematics shifted', weights=weights_clipped_syst, density=True)    


            
            # Set titles and labels
            axes[i].set_title(f'{column}', fontsize=16)
            axes[i].set_xlabel(column)
            axes[i].set_ylabel('Density')
            
            # Add a legend to each subplot
            axes[i].legend()
            del df_clipped_nominal, weights_clipped_nominal, df_clipped_syst, weights_clipped_syst
            gc.collect()
        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.savefig(f'histogram_syst_{self.name}.png')
        plt.show()
        plt.close()

            
    def event_vise_syst(self,df_syst, columns=None, sample_size=100):
        
        df_sample = self.dfall[self.columns].copy().reset_index(drop=True)
        df_sample_syst = df_syst[self.columns].copy().reset_index(drop=True)
        max_sample_size = min(len(df_sample), len(df_sample_syst))
        sample_size = min(sample_size, max_sample_size)
        index = np.random.choice(max_sample_size, sample_size, replace=False)
    #    index = np.random.choice(df_sample_syst.index, sample_size, replace=False)
        df_sample = df_sample.iloc[index]
        df_sample_syst = df_sample_syst.iloc[index]
        df_sample["syst"] = False
        df_sample_syst["syst"] = True
        

        if columns is None:
            columns = self.columns
        sns.set_theme(style="whitegrid")
        
        # Number of rows and columns in the subplot grid
        n_cols = 3  # Number of columns in the subplot grid
        n_rows = int(np.ceil(len(columns) / n_cols))  # Calculate the number of rows needed

        # Create a figure and a grid of subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
        axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy indexing

        for i, column in enumerate(columns):
            field = df_sample[column]
            delta_field = df_sample_syst[column]-df_sample[column]
            axes[i].plot(field,delta_field, 'o', color='blue', label='No Syst')
            axes[i].set_title(f'{column}', fontsize=16)
            axes[i].set_xlabel(column)
            axes[i].set_ylabel('no_syst - syst')
            
            # Add a legend to each subplot
            axes[i].legend()


        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.savefig('event_vise_syst_plot.png')


    def event_vise_syst_arrow(self,df_syst, columns=None, sample_size=100):
        
        df_sample = self.dfall[self.columns].copy().reset_index(drop=True)
        df_sample_syst = df_syst[self.columns].copy().reset_index(drop=True)
        max_sample_size = min(len(df_sample), len(df_sample_syst))
        sample_size = min(sample_size, max_sample_size)
        index = np.random.choice(max_sample_size, sample_size, replace=False)	
  #      index = np.random.choice(df_sample.index, sample_size, replace=False)
        df_sample = df_sample.iloc[index]
        df_sample_syst = df_sample_syst.iloc[index]
        df_sample["syst"] = False
        df_sample_syst["syst"] = True
        
        if columns is None:
            columns = self.columns
        sns.set_theme(style="whitegrid")
        
        # Number of rows and columns in the subplot grid
        n_cols = 3  # Number of columns in the subplot grid
        n_rows = int(np.ceil(len(columns) / n_cols))  # Calculate the number of rows needed

        # Create a figure and a grid of subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
        axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy indexing

        for i, column in enumerate(columns):
            field = df_sample[column]
            delta_field = df_sample_syst[column]-df_sample[column]
            for j in index:
                axes[i].arrow(field[j],field[j],0,delta_field[j],head_width=0.1, head_length=0.1, fc='k', ec='k')
                
            # Adding labels for the arrows
            axes[i].scatter(field, df_sample[column], color='green', label='No syst', zorder=5)
            axes[i].scatter(field, df_sample_syst[column], color='red', label='syst', zorder=5)
            
            axes[i].set_title(f'{column}', fontsize=16)
            axes[i].set_xlabel("column")
            axes[i].set_ylabel(column)
            
            # Add a legend to each subplot
            axes[i].legend()

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.savefig('event_vise_syst_arrow.png')
            

def visualize_scatter(ingestion_result_dict, ground_truth_mus):
    """
    Plots a scatter Plot of ground truth vs. predicted mu values.

    Args:
        * ingestion_result_dict (dict): A dictionary containing the ingestion results.
        * ground_truth_mus (dict): A dictionary of ground truth mu values.
        
    .. Image:: ../images/scatter_plot_mu.png
    """
    plt.figure(figsize=(6, 4))
    for key in ingestion_result_dict.keys():
        ingestion_result = ingestion_result_dict[key]
        mu_hat = np.mean(ingestion_result["mu_hats"])
        mu = ground_truth_mus[key]
        plt.scatter(mu, mu_hat, c='b', marker='o')
    
    plt.xlabel('Ground Truth $\\mu$')
    plt.ylabel('Predicted $\\mu$ (averaged for 100 test sets)')
    plt.title('Ground Truth vs. Predicted $\\mu$ Values')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def roc_curve_wrapper(score, labels, weights, plot_label="model", color="b", lw=2):
    """
    Plots the ROC curve.

    Args:
        * score (ndarray): The score.
        * labels (ndarray): The labels.
        * weights (ndarray): The weights.
        * plot_label (str, optional): The plot label. Defaults to "model".
        * color (str, optional): The color. Defaults to "b".
        * lw (int, optional): The line width. Defaults to 2.
        
    .. Image:: ../images/roc_curve.png
    """

    auc = roc_auc_score(y_true=labels, y_score=score, sample_weight=weights)

    plt.figure(figsize=(8, 7))

    fpr, tpr, _ = roc_curve(y_true=labels, y_score=score, sample_weight=weights)
    plt.plot(fpr, tpr, color=color, lw=lw, label=plot_label + " AUC :" + f"{auc:.3f}")

    plt.plot([0, 1], [0, 1], color="k", lw=lw, linestyle="--")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")

    plt.show()

    plt.close()


def visualize_coverage(ingestion_result_dict, ground_truth_mus):
    """
    Plots a coverage plot of the mu values.

    Args:
        * ingestion_result_dict (dict): A dictionary containing the ingestion results.
        * ground_truth_mus (dict): A dictionary of ground truth mu values.
        
    .. Image:: ../images/coverage_plot.png
    """

    for key in ingestion_result_dict.keys():
        plt.figure( figsize=(5, 5))

        ingestion_result = ingestion_result_dict[key]
        mu = ground_truth_mus[key]
        mu_hats = np.mean(ingestion_result["mu_hats"])
        p16s = ingestion_result["p16"]
        p84s = ingestion_result["p84"]
        
        # plot horizontal lines from p16 to p84
        for i, (p16, p84) in enumerate(zip(p16s, p84s)):
            if i == 0:
                plt.hlines(y=i, xmin=p16, xmax=p84, colors='b', label='Coverage interval')
            else:   
                plt.hlines(y=i, xmin=p16, xmax=p84, colors='b')

        plt.vlines(x=mu_hats, ymin=0, ymax=len(p16s), colors='r', linestyles='dashed', label='Predicted $\\mu$')
        plt.vlines(x=mu, ymin=0, ymax=len(p16s), colors='g', linestyles='dashed', label='Ground Truth $\\mu$')
        plt.xlabel("$\\mu$")
        plt.ylabel('pseudo-experiments')
        plt.title(f'$\\mu$ distribution - Set_{key}')
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        
    plt.show()

def custom_pretty_print(d):
    table_data = []  # To collect data for tabular printing

    for key, value in d.items():
        if isinstance(value, pd.DataFrame):
            
            table_data.append([key, f"DataFrame of shape {value.shape}", "DataFrame"])
        elif isinstance(value, dict):
            # Convert dictionary to list of tuples for tabulate
            str_dict = f"Dictionary with {len(value.keys())} keys"
            table_data.append([key, str_dict, f"{type(value)}"])
        elif isinstance(value, np.ndarray):
            str_np = (f"Array of shape {value.shape}")
            table_data.append([key, str_np, f"{type(value)}"])
        else:
            try: 
                array = np.array(value)
                str_np = (f"Array of shape {array.shape}")
                table_data.append([key, str_np, f"{type(value)}"])
            except:
                try:
                    table_data.append([key, value, type(value)])
                except:
                    table_data.append([key, "Not Available", "Not Available"])
                    
    # Print collected table data if any
    if table_data:

        print(tabulate(table_data, headers=["Key", "Value","Type"], tablefmt='grid'))
