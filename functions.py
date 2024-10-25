import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kdsource as kds
import time
import os
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from scipy.special import rel_entr

# %matplotlib widget


# EN USO:


def calculate_cumulative_histograms(
    df: pd.DataFrame,
    columns: list,  #
    micro_bins: list,  # Number of bins for the cumulative histogram of the micro column
    macro_bins: list,  # Number of bins for the cumulative histogram of the macro (useless if this is the last function)
    type: str = "equal_bins",
) -> tuple:
    """
    Recursively calculate the cumulative histograms for each dimension.

    Parameter:
    - df: DataFrame with the data.
    - columns: List with the names of the columns.
    - micro_bins: List with the number of bins for the cumulative histogram of the micro column.
    - macro_bins: List with the number of bins for the cumulative histogram of the macro column.
    - type: Type of binning to use.

    Return:
    - Tuple with the cumulative histograms, micro bins and macro bins for each dimension.
    """

    # Pareciera que los microgrupos es mejor hacerlos con equal_bins. Sin embargo, aca esta para implementar equal_area en microgrupos:
    # auxiliar, micro = pd.qcut(
    #     df[columns[0]], q=micro_bins[0], labels=False, retbins=True, duplicates='drop'
    # )

    # # Create a DataFrame with the bins and weights
    # df_aux = pd.DataFrame({'bin': auxiliar, 'wgt': df['wgt']})

    # # Group by the bins and sum the weights
    # counts = df_aux.groupby('bin')['wgt'].sum()

    # # Sort the counts by bin index and convert to numpy array
    # counts = counts.sort_index().to_numpy()

    # del auxiliar

    counts, micro = np.histogram(df[columns[0]], bins=micro_bins[0], weights=df["wgt"])

    cumul = np.cumsum(counts)
    total = counts.sum()
    cumul = (
        np.insert(
            np.cumsum(counts) / total, 0, 0
        )  # Insert 0 at the beginning to make the frec acum start by 0 and interpolate further with it
        if total > 0
        else np.zeros(len(counts) + 1)
    )

    cumul_list = [cumul]
    micro_list = [micro]

    if len(columns) == 1:  # Como es iterativo, esto representa el final de una rama
        macro_list = [None]
        return cumul_list, micro_list, macro_list

    if type == "equal_bins":
        _, macro = np.histogram(df[columns[0]], bins=macro_bins[0])

    if type == "equal_area":
        _, macro = pd.qcut(
            df[columns[0]],
            q=macro_bins[0],
            labels=False,
            retbins=True,
            duplicates="drop",
        )

    macro_list = [macro]

    bin_indices = np.digitize(df[columns[0]], macro) - 1

    for i in range(len(columns) - 1):
        cumul_list.append([])
        micro_list.append([])
        macro_list.append([])

    for i in range(len(macro) - 1):
        df_filtered = df[bin_indices == i]

        cumul_aux, micro_aux, macro_aux = calculate_cumulative_histograms(
            df_filtered, columns[1:], micro_bins[1:], macro_bins[1:], type=type
        )

        for j in range(len(cumul_list) - 1):
            cumul_list[j + 1].append(cumul_aux[j])
            micro_list[j + 1].append(micro_aux[j])
            macro_list[j + 1].append(macro_aux[j])

    return cumul_list, micro_list, macro_list


def index_management(data, indices):
    """
    Get the value of a multi-dimensional array using a list of indices. Used in the sample_nd function.

    Parameter:
    - data: Multi-dimensional array.
    - indices: List of indices.

    Return:
    - The value of the array at the specified indices.
    """
    value = data
    for i in indices:
        value = value[i]
    return value


def sample_nd(
    cumul: list, micro: list, macro: list, columns: list, N: int
) -> pd.DataFrame:
    """
    Sample N values from a N-dimensional distribution.

    Parameter:
    - cumul: List with the cumulative histograms for each dimension.
    - micro: List with the micro histograms for each dimension.
    - macro: List with the macro histograms for each dimension.
    - columns: List with the names of the columns.
    - N: Number of samples to generate.

    Return:
    - A DataFrame with the sampled values.
    """
    sampled_values = []

    if len(columns) == 6:
        for _ in range(N):
            # First dimension
            sampled_0 = np.interp(np.random.rand(), cumul[0], micro[0])
            index_0 = np.searchsorted(macro[0], sampled_0) - 1

            # Second dimension
            sampled_1 = np.interp(
                np.random.rand(), cumul[1][index_0], micro[1][index_0]
            )
            index_1 = np.searchsorted(macro[1][index_0], sampled_1) - 1

            # Third dimension
            sampled_2 = np.interp(
                np.random.rand(), cumul[2][index_0][index_1], micro[2][index_0][index_1]
            )
            index_2 = np.searchsorted(macro[2][index_0][index_1], sampled_2) - 1

            # Fourth dimension
            sampled_3 = np.interp(
                np.random.rand(),
                cumul[3][index_0][index_1][index_2],
                micro[3][index_0][index_1][index_2],
            )
            index_3 = (
                np.searchsorted(macro[3][index_0][index_1][index_2], sampled_3) - 1
            )

            # Fifth dimension
            sampled_4 = np.interp(
                np.random.rand(),
                cumul[4][index_0][index_1][index_2][index_3],
                micro[4][index_0][index_1][index_2][index_3],
            )
            index_4 = (
                np.searchsorted(macro[4][index_0][index_1][index_2][index_3], sampled_4)
                - 1
            )

            # Sixth dimension
            sampled_5 = np.interp(
                np.random.rand(),
                cumul[5][index_0][index_1][index_2][index_3][index_4],
                micro[5][index_0][index_1][index_2][index_3][index_4],
            )

            # Append the sampled values
            sampled_values.append(
                [sampled_0, sampled_1, sampled_2, sampled_3, sampled_4, sampled_5]
            )

        return pd.DataFrame(sampled_values, columns=columns)

    if len(columns) == 5:
        for _ in range(N):
            # First dimension
            sampled_0 = np.interp(np.random.rand(), cumul[0], micro[0])
            index_0 = np.searchsorted(macro[0], sampled_0) - 1

            # Second dimension
            sampled_1 = np.interp(
                np.random.rand(), cumul[1][index_0], micro[1][index_0]
            )
            index_1 = np.searchsorted(macro[1][index_0], sampled_1) - 1

            # Third dimension
            sampled_2 = np.interp(
                np.random.rand(), cumul[2][index_0][index_1], micro[2][index_0][index_1]
            )
            index_2 = np.searchsorted(macro[2][index_0][index_1], sampled_2) - 1

            # Fourth dimension
            sampled_3 = np.interp(
                np.random.rand(),
                cumul[3][index_0][index_1][index_2],
                micro[3][index_0][index_1][index_2],
            )
            index_3 = (
                np.searchsorted(macro[3][index_0][index_1][index_2], sampled_3) - 1
            )

            # Fifth dimension
            sampled_4 = np.interp(
                np.random.rand(),
                cumul[4][index_0][index_1][index_2][index_3],
                micro[4][index_0][index_1][index_2][index_3],
            )

            # Append the sampled values
            sampled_values.append(
                [sampled_0, sampled_1, sampled_2, sampled_3, sampled_4]
            )

        return pd.DataFrame(sampled_values, columns=columns)

    # General case
    for _ in range(N):
        sample = []
        index = []

        for dim in range(len(columns)):
            cumul_aux = index_management(cumul[dim], index)
            micro_aux = index_management(micro[dim], index)

            # Sample the value for the current dimension
            sampled_value = np.interp(np.random.rand(), cumul_aux, micro_aux)
            sample.append(sampled_value)

            if (
                dim < len(columns) - 1
            ):  # Check if there are still more dimensions to process
                macro_aux = index_management(macro[dim], index)
                auxiliar = np.searchsorted(macro_aux, sampled_value) - 1
                index.append(auxiliar)

        sampled_values.append(sample)

    return pd.DataFrame(sampled_values, columns=columns)


def plot_correlated_variables(
    df: pd.DataFrame,
    columns: list,  # Por ahora todos tienen los mismos bines
    bin_size_diagonal: int = 100,
    bin_size_off_diagonal: int = 100,
    save: bool = False,
    plot: bool = True,
    density: bool = True,
    weight=True,
    filename: str = "correlated_histograms.png",
) -> None:

    fig, axes = plt.subplots(nrows=len(columns), ncols=len(columns), figsize=(15, 15))

    # Iterate through the rows and columns
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i == j:
                # Diagonal plots: Plot histograms when col1 == col2
                if density:
                    df[col1].plot(
                        kind="hist",
                        ax=axes[i, j],
                        bins=bin_size_diagonal,
                        color="skyblue",
                        title=f"{col1}",
                        density=True,
                    )
                else:
                    df[col1].plot(
                        kind="hist",
                        ax=axes[i, j],
                        bins=bin_size_diagonal,
                        color="skyblue",
                        title=f"{col1}",
                    )
            else:
                # Off-diagonal plots: 2D histograms (heatmap-like) for col1 vs col2
                axes[i, j].hist2d(
                    df[col2], df[col1], bins=bin_size_off_diagonal, cmap="Blues"
                )
                axes[i, j].set_title(f"{col1} vs {col2}")
            # Set labels
            if i == len(columns) - 1:
                axes[i, j].set_xlabel(col2)
            if j == 0:
                axes[i, j].set_ylabel(col1)

    plt.tight_layout()  # Adjust layout to prevent overlap

    # Save the figure as a PNG file
    if save:
        plt.savefig(filename, dpi=300)

    if plot:
        plt.show()


def plot_correlated_variables_counts(
    counts_1d: np.ndarray,
    counts_2d: np.ndarray,
    edges_1d: list,
    edges_2d_1: list,
    edges_2d_2: list,
    columns: list,  
    filename: str = "correlated_histograms.png",
) -> None:

    fig, axes = plt.subplots(nrows=len(columns), ncols=len(columns), figsize=(15, 15))

    # Iterate through the rows and columns
    iterator = 0
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i == j:
                # Diagonal plots: Plot histograms when col1 == col2
                axes[i, j].stairs(counts_1d[i], edges_1d[i], fill=True, color="skyblue")
                axes[i, j].set_title(f"{col1}")
            if j > i:
                axes[i, j].pcolormesh(
                    edges_2d_2[iterator],
                    edges_2d_1[iterator],
                    counts_2d[iterator],
                    cmap="Blues",
                )
                axes[i, j].set_title(f"{col1} vs {col2}")
                axes[j, i].pcolormesh(
                    edges_2d_1[iterator],
                    edges_2d_2[iterator],
                    counts_2d[iterator].T,
                    cmap="Blues",
                )
                axes[j, i].set_title(f"{col2} vs {col1}")
                iterator += 1
                
            # Set labels
            if i == len(columns) - 1:
                axes[i, j].set_xlabel(col2)
            if j == 0:
                axes[i, j].set_ylabel(col1)

    plt.tight_layout()  # Adjust layout to prevent overlap

    # Save the figure as a PNG file
    plt.savefig(filename, dpi=300)

    # Close the figure to free up memory
    plt.close(fig)


def get_time_and_counts(
    columns: list,
    cumul: list,
    micro: list,
    macro: list,
    N: int,
    edges_1d: list,
    edges_2d_1: list,
    edges_2d_2: list,
) -> tuple:
    """
    Calculate the time needed to calculate the cumulative histograms and to sample the data.
    Also calculate the counts of the sampled data for each column.

    Parameter:
    - df: DataFrame with the data.
    - columns: List with the names of the columns.
    - micro_bins: List with the number of bins for the cumulative histogram of the micro column.
    - macro_bins: List with the number of bins for the cumulative histogram of the macro column.
    - N: Number of samples to generate.
    - edges: List with the edges of the bins for each column to calculate the counts based on the original histogram.
    - type: Type of binning to use.

    Return:
    - Tuple with the time needed to calculate the cumulative histograms and to sample the data, and the counts of the sampled data for each column.
    """

    start_time = time.perf_counter()
    df_sampled = sample_nd(cumul, micro, macro, columns, N)
    end_time = time.perf_counter()
    time_sampling = end_time - start_time

    counts_1d = np.array(
        [
            np.histogram(df_sampled[column], bins=edge)[0]
            for column, edge in zip(columns, edges_1d)
        ]
    )

    counts_2d = np.array(
        [
            np.histogram2d(
                df_sampled[columns[i]], df_sampled[columns[j]], bins=[edges1, edges2]
            )[0]
            for (i, j), edges1, edges2 in zip(
                [
                    (i, j)
                    for i in range(len(columns) - 1)
                    for j in range(i + 1, len(columns))
                ],
                edges_2d_1,
                edges_2d_2,
            )
        ]
    )

    return time_sampling, counts_1d, counts_2d


def barrido(
    columns_order: list,
    micro_bins: list,
    macro_bins: list,
    N: list,
    # batches: int,
    type: list,
    df: pd.DataFrame,
    save: bool = False,
) -> None:
    """
    Function to run a series of tests with different number of samples and store the results in a list of lists to later plot them.

    Parameters
    ----------
    columns_order : list
        List with the names of the columns to be used in the tests.
    micro_bins : list
        List with the number of micro bins for each column.
    macro_bins : list
        List with the number of macro bins for each column.
    N : list
        List with the number of samples to be used in the tests.
    batches : int
        Number of batches to be used in the tests.
    type : str
        Type of binning to be used in the tests.
    file : str
        Path to the surface source file.
    save : bool, optional
        If True, the plots and the results will be saved in a folder. The default is False.

    Returns
    -------
    time_histo : list
        List with the time to calculate the histograms for each number of samples.
    time_sample : list
        List with the time to sample the data for each number of samples.
    kl_divergence_1d : list
        List with the KL divergence for the 1d histograms for each number of samples.
    kl_divergence_2d : list
        List with the KL divergence for the 2d histograms for each number of samples.
    """

    min_1d, max_1d, min_2d, max_2d = [], [], [], [] 

    for index, columns_order, micro_bins, macro_bins, N, type in zip(
        range(len(columns_order)),
        columns_order,
        micro_bins,
        macro_bins,
        N,
        type,
    ):
        print(f"Index: {index+1}")
        # Calculate the original histograms counts
        counts_1d_original, edges_1d, counts_2d_original, edges_2d_1, edges_2d_2 = (
            calculate_original_histograms(df, columns_order, save=save, index=index + 1)
        )

        # Initialize variables to store the results. The first element is the name of the column to later use in the plots
        time_histo = ["Tiempo de calculo de los histogramas"]
        time_sample = ["Tiempo de muestreo"]
        kl_divergence_1d = [[column] for column in columns_order]
        kl_divergence_2d = [
            [(columns_order[i], columns_order[j])]
            for i in range(len(columns_order) - 1)
            for j in range(i + 1, len(columns_order))
        ]

        # Calculate the histograms for the original data
        start_time = time.perf_counter()
        cumul, micro, macro = calculate_cumulative_histograms(
            df, columns_order, micro_bins, macro_bins, type=type
        )
        end_time = time.perf_counter()
        time_histo.extend([end_time - start_time] * len(N))

        # Initialize variables to store the results of the batches
        (
            time_sample_container,
            counts_1d_container,
            counts_2d_container,
        ) = (0, np.zeros_like(counts_1d_original), np.zeros_like(counts_2d_original))

        # Run the tests
        for j, n in enumerate(N):
            print("Number of samples: ", sum(N[: j + 1]))

            # # Run the batches
            # for batch in range(batches):
            # print("Batch: ", batch + 1)

            (
                time_sample_batch,
                counts_1d_batch,
                counts_2d_batch,
            ) = get_time_and_counts(
                columns_order,
                cumul,
                micro,
                macro,
                n,
                edges_1d,
                edges_2d_1,
                edges_2d_2,
            )
            time_sample_container += time_sample_batch
            counts_1d_container += counts_1d_batch
            counts_2d_container += counts_2d_batch

            # normalize counts_container and take aways the zeros
            counts_1d_container_normalized = np.array(
                [counts / counts.sum() for counts in counts_1d_container]
            )
            counts_1d_container_normalized[counts_1d_container_normalized == 0] = 1e-6

            counts_2d_container_normalized = np.array(
                [counts / counts.sum() for counts in counts_2d_container]
            )
            counts_2d_container_normalized[counts_2d_container_normalized == 0] = 1e-6

            # Save plots of variables for each number of samples
            if save:
                if not os.path.exists(f"{index+1}"):
                    os.makedirs(f"{index+1}")
                plot_correlated_variables_counts(
                    counts_1d=counts_1d_container_normalized,
                    counts_2d=counts_2d_container_normalized,
                    edges_1d=edges_1d,
                    edges_2d_1=edges_2d_1,
                    edges_2d_2=edges_2d_2,
                    columns=columns_order,
                    filename=f"{index+1}/{sum(N[:j+1])}.png",
                )

            # Calculate the KL divergence for the histograms
            for i, divergence in enumerate(
                [
                    sum(rel_entr(original, synthetic))
                    for original, synthetic in zip(
                        counts_1d_original, counts_1d_container_normalized
                    )
                ]
            ):
                kl_divergence_1d[i].append(divergence)

            for i, divergence in enumerate(
                [
                    np.sum(rel_entr(original, synthetic))
                    for original, synthetic in zip(
                        counts_2d_original, counts_2d_container_normalized
                    )
                ]
            ):
                kl_divergence_2d[i].append(divergence)

            # Append the time results
            time_sample.append(time_sample_container)

        # Obtain min and max for 1d and 2d divergences
        kl_divergence_1d_T = list(zip(*kl_divergence_1d))[-1]
        min_1d.append(np.min(kl_divergence_1d_T))
        max_1d.append(np.max(kl_divergence_1d_T))

        kl_divergence_2d_T = list(zip(*kl_divergence_2d))[-1]
        min_2d.append(np.min(kl_divergence_2d_T))
        max_2d.append(np.max(kl_divergence_2d_T))


        # Plot the results of the tests
        if not os.path.exists(f"{index+1}"):
            os.makedirs(f"{index+1}")
        plot_results_barrido(
            N,
            time_histo,
            time_sample,
            kl_divergence_1d,
            kl_divergence_2d,
            columns_order,
            micro_bins,
            macro_bins,
            type,
            name=f"{index+1}/results.png",
        )

    # Save the results in the .csv file reading it from the file and turning it into a dataframe
    data = pd.read_csv("index_information.csv", index_col=0)
    data["min_1d"] = min_1d
    data["max_1d"] = max_1d
    data["min_2d"] = min_2d
    data["max_2d"] = max_2d
    data.to_csv("index_information.csv", index=True)


def plot_results_barrido(
    N: list,
    time_histo: list,
    time_sample: list,
    kl_divergence_1d: list,
    kl_divergence_2d: list,
    columns_order: list,
    micro_bins: list,
    macro_bins: list,
    type: str,
    name: str = None,
) -> None:
    """
    Plot the results of the tests.

    Parameters
    ----------
    N : list
        List with the number of samples used in the tests.
    batches : int
        Number of batches used in the tests.
    time_histo : list
        List with the time to calculate the histograms for each number of samples.
    time_sample : list
        List with the time to sample the data for each number of samples.
    kl_divergence_1d : list
        List with the KL divergence for the 1d histograms for each number of samples.
    kl_divergence_2d : list
        List with the KL divergence for the 2d histograms for each number of samples.
    columns_order : list
        List with the names of the columns used in the tests.
    micro_bins : list
        List with the number of micro bins used in the tests.
    macro_bins : list
        List with the number of macro bins used in the tests.
    type : str
        Type of binning used in the tests.
    """

    N_plot = [sum(N[: i + 1]) for i, _ in enumerate(N)]

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Plot 1: Time for histograms
    axes[0, 0].plot(N_plot, time_histo[1:], label=time_histo[0], marker="o")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("Numero de muestras")
    axes[0, 0].set_ylabel("Tiempo [s]")
    axes[0, 0].legend()
    axes[0, 0].grid()
    axes[0, 0].set_ylim(bottom=0)
    axes[0, 0].set_title(
        "Tiempo de calculo en funcion del numero de muestras", fontsize=10
    )

    # Plot 2: Time for sampling
    axes[0, 1].loglog(N_plot, time_sample[1:], label=time_sample[0], marker="o")
    axes[0, 1].set_xlabel("Numero de muestras")
    axes[0, 1].set_ylabel("Tiempo [s]")
    axes[0, 1].legend()
    axes[0, 1].grid()
    axes[0, 0].set_ylim(bottom=1)
    axes[0, 1].set_title(
        "Tiempo de muestreo en funcion del numero de muestras", fontsize=10
    )

    # Plot 3: KL divergence_1d
    for kl in kl_divergence_1d:
        axes[1, 0].loglog(N_plot, kl[1:], label=kl[0], marker="o")
    axes[1, 0].set_xlabel("Numero de muestras")
    axes[1, 0].set_ylabel("KL divergence 1D")
    axes[1, 0].legend()
    axes[1, 0].grid()
    axes[1, 0].set_title(
        "Divergencia KL 1D en funcion del numero de muestras", fontsize=10
    )

    # Plot 4: KL divergence_2d
    for kl in kl_divergence_2d:
        axes[1, 1].loglog(N_plot, kl[1:], label=str(kl[0]), marker="o")
    axes[1, 1].set_xlabel("Numero de muestras")
    axes[1, 1].set_ylabel("KL divergence 2D")
    axes[1, 1].legend(fontsize="small")
    axes[1, 1].grid()
    axes[1, 1].set_title(
        "Divergencia KL 2D en funcion del numero de muestras", fontsize=10
    )

    # Add title with information of the run
    fig.suptitle(
        "Columns order: "
        + str(columns_order)
        + "\n"
        + "Micro bins: "
        + str(micro_bins)
        + "\n"
        + "Macro bins: "
        + str(macro_bins)
        + "\n"
        + "Type: "
        + type
    )

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure as a PNG file
    if name is not None:
        plt.savefig(name, dpi=300)

    # Close the figure to free up memory
    plt.close(fig)


def manage_n_samples(
    amount: int,
    max_batch_size: float = 1e7,
    min_batch_size: float = 1e3,  # use base 10 for batch_size
) -> np.ndarray:
    """
    Calculate the number of samples to generate for each batch.

    Parameters
    ----------
    amount : int
        Total number of samples to generate.
    max_batch_size : float, optional
        Maximum number of samples to generate in a batch. The default is 1e7.
    min_batch_size : float, optional
        Minimum number of samples to generate in a batch. The default is 1e3.

    Returns
    -------
    n : np.ndarray[int]
        Array with the number of samples to generate for each batch.
    """
    if amount < min_batch_size:
        return np.array([amount]).astype(int)
    excess = amount / max_batch_size

    n = np.logspace(
        np.log10(min_batch_size),
        np.log10(max_batch_size),
        1 + int(np.log10(max_batch_size / min_batch_size)) * 2,
    )

    n = n.astype(int)

    if excess >= 1:
        n[1:] = n[1:] - n[:-1]
        # print((excess-int(excess))*max_batch_size)
        # print(0.00001 * max_batch_size)
        n = np.append(n, [max_batch_size] * (int(excess) - 1))
        if excess - int(excess) > 0:
            n = np.append(n, round((excess - int(excess)) * max_batch_size, 0))
        n = n.astype(int)
    else:
        n = n[n <= amount]
        if n[-1] != amount:
            n = np.append(n, amount)
        n[1:] = n[1:] - n[:-1]
        n = n.astype(int)

    return n


def save_information(
    type: list, N_max: list, columns_order: list, micro_bins: list, macro_bins: list
) -> None:
    """
    Save the information of the tests in a CSV file.

    Parameters
    ----------
    type : list
        List with the type of binning used in the tests.
    N_max : list
        List with the maximum number of samples used in the tests.
    columns_order : list
        List with the names of the columns used in the tests.
    micro_bins : list
        List with the number of micro bins used in the tests.
    macro_bins : list
        List with the number of macro bins used in the tests.
    """
    data = {
        "type": type,
        "N_max": N_max,
        "columns_order": columns_order,
        "micro_bins": micro_bins,
        "macro_bins": macro_bins,
    }
    df = pd.DataFrame(data)

    # Set the index to start from 1
    df.index = df.index + 1

    df.to_csv("index_information.csv", index=True)


def calculate_original_histograms(
    df: pd.DataFrame,
    columns_order: list,
    bin_size: int = 100,
    save: bool = False,
    index: int = None,
) -> list[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the original histograms for the data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data.
    columns_order : list
        List with the names of the columns to be used in the tests.
    bin_size : int, optional
        Number of bins to use in the histograms. The default is 100.
    save : bool, optional
        If True, the plots of the original histograms will be saved. The default is False.

    Returns
    -------
    counts_1d_original : np.ndarray
        Array with the counts of the 1d histograms for each column.
    edges_1d : np.ndarray
        Array with the edges of the bins for the 1d histograms.
    counts_2d_original : np.ndarray
        Array with the counts of the 2d histograms for each pair of columns.
    edges_2d_1 : np.ndarray
        Array with the edges of the bins for the first dimension of the 2d histograms.
    edges_2d_2 : np.ndarray
        Array with the edges of the bins for the second dimension of the 2d histograms.
    """
    # Calculate the original histogram 1d
    counts_1d_original, edges_1d = zip(
        *[
            counts_edges
            for counts_edges in [
                np.histogram(df[column], bins=bin_size, weights=df["wgt"])
                for column in columns_order
            ]
        ]
    )
    counts_1d_original, edges_1d = np.array(counts_1d_original), np.array(edges_1d)
    counts_1d_original = np.array(
        [counts / counts.sum() for counts in counts_1d_original]
    )
    counts_1d_original[counts_1d_original == 0] = 1e-6

    # Calculate the original histogram 2d
    counts_2d_original, edges_2d_1, edges_2d_2 = zip(
        *[
            np.histogram2d(
                df[columns_order[i]],
                df[columns_order[j]],
                bins=bin_size,
                weights=df["wgt"],
            )
            for i in range(len(columns_order) - 1)
            for j in range(i + 1, len(columns_order))
        ]
    )

    counts_2d_original, edges_2d_1, edges_2d_2 = (
        np.array(counts_2d_original),
        np.array(edges_2d_1),
        np.array(edges_2d_2),
    )
    counts_2d_original = np.array(
        [counts / counts.sum() for counts in counts_2d_original]
    )
    counts_2d_original[counts_2d_original == 0] = 1e-6

    # Save plots of variables for the original data
    if save:
        if not os.path.exists(f"{index}") and index is not None:
            os.makedirs(f"{index}")
        plot_correlated_variables_counts(
            counts_1d=counts_1d_original,
            counts_2d=counts_2d_original,
            edges_1d=edges_1d,
            edges_2d_1=edges_2d_1,
            edges_2d_2=edges_2d_2,
            columns=columns_order,
            filename=f"original.png" if index is None else f"{index}/original.png",
        )

    return counts_1d_original, edges_1d, counts_2d_original, edges_2d_1, edges_2d_2


def barrido_combinations(
    columns_order: list, micro_bins: list, macro_bins: list, N_max: list, type: list
):
    columns_order_aux = []
    micro_bins_aux = []
    macro_bins_aux = []
    N_max_aux = []
    type_aux = []

    for columns_order_ in columns_order:
        for micro_bins_ in micro_bins:
            for macro_bins_ in macro_bins:
                for N_max_ in N_max:
                    for type_ in type:
                        columns_order_aux.append(columns_order_)
                        micro_bins_aux.append(micro_bins_)
                        macro_bins_aux.append(macro_bins_)
                        N_max_aux.append(manage_n_samples(N_max_))
                        type_aux.append(type_)
    return columns_order_aux, micro_bins_aux, macro_bins_aux, N_max_aux, type_aux


def comparacion_barrido(data_file: str = "index_information.csv", result_name:str = "comparacion.png")-> None:

    data = pd.read_csv(data_file, index_col=0)

    # Create a figure with two subplots horizontally
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    # Prepare data for variable 1
    for i in range(len(data)):
        if i % 2 == 0:
            ax1.plot([i, i], [data["min_1d"].iloc[i], data["max_1d"].iloc[i]], color='blue', lw=2, marker='o')  # Plot the box from min to max
        else:
            ax1.plot([i, i], [data["min_1d"].iloc[i], data["max_1d"].iloc[i]], color='red', lw=2, marker='o')
    ax1.set_xticks(np.arange(len(data)))
    ax1.set_xticklabels(data.index,rotation=90)
    ax1.grid()
    ax1.set_yscale('log')
    ax1.set_title('Divergencia_kl_1d (Min to Max)')
    ax1.set_xlabel('Index')


    # Prepare data for variable 2
    for i in range(len(data)):
        if i % 2 == 0:
            ax2.plot([i, i], [data["min_2d"].iloc[i], data["max_2d"].iloc[i]], color='blue', lw=2, marker='o')  # Plot the box from min to max
        else:
            ax2.plot([i, i], [data["min_2d"].iloc[i], data["max_2d"].iloc[i]], color='red', lw=2, marker='o')
    ax2.set_xticks(np.arange(len(data)))
    ax2.set_xticklabels(data.index,rotation=90)
    ax2.grid()
    ax2.set_yscale('log')
    ax2.set_title('Divergencia_kl_2d (Min to Max)')
    ax2.set_xlabel('Index')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure as a PNG file
    plt.savefig(result_name, dpi=300)

    # Close the figure to free up memory
    plt.close(fig)











# EN DESUSO:


def calculate_cumulative_histograms2(
    df: pd.DataFrame,
    columns: list,  #
    micro_bins: list,  # Number of bins for the cumulative histogram of the micro column
    macro_bins: list,  # Number of bins for the cumulative histogram of the macro4 (useless if this is the last function)
    type: str = "equal_bins",
) -> tuple:
    """
    Recursively calculate the cumulative histograms for each dimension.

    Parameter:
    - df: DataFrame with the data.
    - columns: List with the names of the columns.
    - micro_bins: List with the number of bins for the cumulative histogram of the micro column.
    - macro_bins: List with the number of bins for the cumulative histogram of the macro column.
    - type: Type of binning to use.

    Return:
    - Tuple with the cumulative histograms, micro bins and macro bins for each dimension.
    """

    cumul_list = []
    micro_list = []
    macro_list = []

    if type == "equal_bins":
        counts, micro = np.histogram(df[columns[0]], bins=micro_bins[0])

    if type == "equal_area":
        auxiliar, micro = pd.qcut(
            df[columns[0]], q=micro_bins[0], labels=False, retbins=True
        )
        counts = auxiliar.value_counts().sort_index().to_numpy()
        del auxiliar

    cumul = np.cumsum(counts)
    total = counts.sum()
    cumul = (
        np.insert(np.cumsum(counts) / total, 0, 0)
        if total > 0
        else np.zeros(len(counts) + 1)
    )
    # if total:
    #     cumul = cumul / total
    # cumul = np.insert(
    #     cumul, 0, 0
    # )  # Insert 0 at the beginning to make the frec acum start by 0 and interpolate further with it

    cumul_list.append(cumul)
    micro_list.append(micro)

    if len(columns) == 1:
        macro_list.append(None)
        return cumul_list, micro_list, macro_list

    if type == "equal_bins":
        _, macro = np.histogram(df[columns[0]], bins=macro_bins[0])

    if type == "equal_area":
        _, macro = pd.qcut(df[columns[0]], q=macro_bins[0], labels=False, retbins=True)

    macro_list.append(macro)

    bin_indices = np.digitize(df[columns[0]], macro) - 1

    for i in range(len(columns) - 1):
        cumul_list.append([])
        micro_list.append([])
        macro_list.append([])

    for i in range(len(macro) - 1):
        # df_filtered = df.loc[
        #     (df[columns[0]] >= macro[i]) & (df[columns[0]] <= macro[i + 1])
        # ]
        mask = bin_indices == i
        df_filtered = df[mask]

        cumul_aux, micro_aux, macro_aux = calculate_cumulative_histograms(
            df_filtered,
            columns[1:],
            micro_bins[1:],
            macro_bins[1:],
        )

        for j in range(len(cumul_list) - 1):
            cumul_list[j + 1].append(cumul_aux[j])
            micro_list[j + 1].append(micro_aux[j])
            macro_list[j + 1].append(macro_aux[j])

    return cumul_list, micro_list, macro_list


def change_zeros(array: np.ndarray) -> None:
    for i in range(len(array)):
        if array[i] == 0:
            array[i] = 1e-6


class histograms:
    def __init__(self, data: pd.Series, bins: int) -> None:
        """ """
        self.data = data  # (pd.Series)
        self.bins = bins  # (int)
        self.bin_edges = None  # (np.ndarray) Values of the edges of the bins
        self.counts = None  # (np.ndarray) Number of counts in each bin
        self.bin_centers = None  # Still not used
        self.cumulative = None  # (np.ndarray) Cumulative histogram and starts with 0
        self.bin_edges_equal_area = None
        self.counts_equal_area = None

    def calculate_histogram(self) -> None:
        """
        Calculate the histogram of the data.
        """
        if self.data is not None:
            self.counts, self.bin_edges = np.histogram(self.data, bins=self.bins)
        else:
            raise ValueError("Data is not provided")

    def calculate_histogram_cumulative(self) -> None:
        """
        Calculate the cumulative histogram.
        It inserts a 0 at the beginning to make the frec acum start by 0.
        """
        self.cumulative = np.cumsum(self.counts) / self.counts.sum()
        self.cumulative = np.insert(self.cumulative, 0, 0)
        # Insert 0 at the beginning to make the frec acum start by
        # 0 and interpolate further with it

    def calculate_histogram_equal_area(self, bins: int = None) -> None:
        """
        Calculate the histogram with equal area.
        """
        if bins is None:
            bins = round(1 + 3.322 * np.log10(len(self.data)))

        aux = np.linspace(0, len(self.data) - 1, bins + 1)

        for i in range(len(aux)):
            aux[i] = round(aux[i])

        self.bin_edges_equal_area = self.data[aux]
        self.counts_equal_area, _ = np.histogram(
            self.data, bins=self.bin_edges_equal_area
        )

    def calculate_histogram_new(self, bins: int) -> None:
        # Tal vez no hace falta. Tal vez podria tener un mejor nombre.
        """
        Calculate the histogram with new bins.
        """
        self.bins_new = bins
        self.counts_new, self.bin_edges_new = np.histogram(self.data, bins=bins)

    # Plotting functions

    def plot_histogram(self, density=True, figure=True, alpha: float = 1) -> None:
        # Podria tener un parametro para agregar una leyenda
        """
        Plot the histogram.
        Figure: Create a new figure.
        Alpha: Transparency of the bars. 1 is opaque, 0 is transparent.
        """
        if self.bin_edges is not None and self.counts is not None:
            if figure:
                plt.figure()
            plt.hist(
                self.bin_edges[:-1],
                bins=self.bin_edges,
                weights=self.counts,
                edgecolor="black",
                density=density,
                label="Histogram - Bins: " + str(len(self.counts)),
                alpha=alpha,
            )
            plt.xlabel("Value")
            plt.ylabel("Density" if density else "Counts")
            plt.grid()
            plt.legend()
            plt.title("Histogram with Density" if density else "Histogram")
        else:
            raise ValueError("Histogram counts and bin edges are not calculated")

    def plot_cumulative_histogram(self, figure=True) -> None:
        """
        Plot the cumulative histogram.
        Figure: Create a new figure.
        """
        if figure:
            plt.figure()
        plt.plot(self.bin_edges, self.cumulative, label="Cumulative Histogram")
        plt.xlabel("Value")
        plt.ylabel("Cumulative Frequency")
        plt.grid()
        plt.legend()
        plt.title("Cumulative Histogram")

    def plot_histogram_equal_area(
        self, density=True, figure=True, alpha: float = 1
    ) -> None:
        """
        Plot the histogram with equal area.
        Figure: Create a new figure.
        Alpha: Transparency of the bars. 1 is opaque, 0 is transparent.
        """
        if self.bin_edges_equal_area is not None and self.counts_equal_area is not None:
            if figure:
                plt.figure()
            plt.hist(
                self.bin_edges_equal_area[:-1],
                bins=self.bin_edges_equal_area,
                weights=self.counts_equal_area,
                edgecolor="black",
                density=density,
                label="Equal Area Histogram - Bins: "
                + str(len(self.counts_equal_area)),
                alpha=alpha,
            )
            plt.xlabel("Value")
            plt.ylabel("Density" if density else "Counts")
            plt.grid()
            plt.legend()
            plt.title(
                "Equal Area Histogram with Density"
                if density
                else "Equal Area Histogram"
            )
        else:
            raise ValueError(
                "Equal area histogram counts and bin edges are not calculated"
            )

    def plot_histogram_new(self, density=True, figure=True, alpha: float = 1) -> None:
        if self.bin_edges_new is not None and self.counts_new is not None:
            if figure:
                plt.figure()
            plt.hist(
                self.bin_edges_new[:-1],
                bins=self.bin_edges_new,
                weights=self.counts_new,
                edgecolor="black",
                density=density,
                label="Histogram - Bins: " + str(len(self.counts_new)),
                alpha=alpha,
            )
            plt.xlabel("Value")
            plt.ylabel("Density" if density else "Counts")
            plt.grid()
            plt.legend()
            plt.title("Histogram with Density" if density else "Histogram")
        else:
            raise ValueError("Histogram counts and bin edges are not calculated")


def set_Y_values(x: float) -> float:
    return np.random.normal(3, np.abs(10 - np.abs(x)) / 4)


def delete_data(
    df: pd.DataFrame, x_min: float, x_max: float, y_min: float, y_max: float
):
    # Use inplace modification to filter the DataFrame
    df.drop(
        df[
            (df["X"] < x_min)
            | (df["X"] > x_max)
            | (df["Y"] < y_min)
            | (df["Y"] > y_max)
        ].index,
        inplace=True,
    )


def plot_hist2d(
    x: pd.Series,
    y: pd.Series,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    figure=True,
    title: str = "2D Histogram",
    xlabel: str = "X",
    ylabel: str = "Y",
) -> None:
    """
    Plot a 2D histogram.
    """
    if figure:
        plt.figure()
    plt.hist2d(x, y, bins=[x_edges, y_edges], cmap="viridis")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label="Counts")
    plt.title(title)


def get_cumulatives(
    histo1: histograms,
    df: pd.DataFrame,
    type: str = "new",
    column1: str = "X",
    column2: str = "Y",
    plot=False,
) -> tuple:
    cumulatives = []
    edges = []

    if type == "new":
        for i in range(len(histo1.counts_new)):
            df_filtered = df[
                (df[column1] >= histo1.bin_edges_new[i])
                & (df[column1] <= histo1.bin_edges_new[i + 1])
            ]

            # # Sort the data based on the y column
            # df_filtered.sort_values(column2, inplace=True)
            # df_filtered.reset_index(drop=True, inplace=True)

            histo2 = histograms(data=df_filtered[column2], bins=75)
            histo2.calculate_histogram()
            if plot:
                histo2.plot_histogram(density=True, figure=False, alpha=0.5)
            histo2.calculate_histogram_cumulative()
            cumulatives.append(histo2.cumulative)
            edges.append(histo2.bin_edges)

    return cumulatives, edges


def sample(
    cumulative1: np.ndarray,
    edges1: np.ndarray,
    macro_edges1: np.ndarray,
    cumulative2,
    edges2,
    amount: int = 1,
) -> tuple:

    # plot the sample
    sampled_values_1 = []
    sampled_values_2 = []
    for i in range(amount):
        random_number = np.random.rand()
        sample_value_1 = np.interp(random_number, cumulative1, edges1)
        auxiliar = np.searchsorted(macro_edges1, sample_value_1) - 1
        random_number = np.random.rand()
        sampled_value2 = np.interp(
            random_number, cumulative2[auxiliar], edges2[auxiliar]
        )
        sampled_values_1.append(sample_value_1)
        sampled_values_2.append(sampled_value2)

    return sampled_values_1, sampled_values_2


def sample_3d(
    cumulative1: np.ndarray,
    edges1: np.ndarray,
    macro_edges1: np.ndarray,
    cumulative2,
    edges2,
    macro_edges2: np.ndarray,
    cumulative3,
    edges3,
    amount: int = 1,
) -> tuple:

    # plot the sample
    sampled_values_1 = []
    sampled_values_2 = []
    sampled_values_3 = []
    for i in range(amount):
        random_number = np.random.rand()
        sample_value_1 = np.interp(random_number, cumulative1, edges1)
        auxiliar1 = np.searchsorted(macro_edges1, sample_value_1) - 1
        random_number = np.random.rand()
        sampled_value2 = np.interp(
            random_number, cumulative2[auxiliar1], edges2[auxiliar1]
        )
        auxiliar2 = np.searchsorted(macro_edges2[auxiliar1], sampled_value2) - 1
        random_number = np.random.rand()
        sampled_value3 = np.interp(
            random_number,
            cumulative3[auxiliar1][auxiliar2],
            edges3[auxiliar1][auxiliar2],
        )
        sampled_values_1.append(sample_value_1)
        sampled_values_2.append(sampled_value2)
        sampled_values_3.append(sampled_value3)

    return sampled_values_1, sampled_values_2, sampled_values_3


def sample_recur2(
    cumul: list,
    micro: list,
    macro: list,
    columns: list,
    sample: list,
    index: list = None,
) -> None:
    if index is None:
        index = []

    cumul_aux = index_management(cumul[len(index)], index)
    micro_aux = index_management(micro[len(index)], index)

    sampled_value = np.interp(np.random.rand(), cumul_aux, micro_aux)
    sample.append(sampled_value)

    if len(columns) > 1:
        macro_aux = index_management(macro[len(index)], index)
        auxiliar = np.searchsorted(macro_aux, sampled_value) - 1
        index.append(auxiliar)
        sample_recur2(cumul, micro, macro, columns[1:], sample, index)


def sample_df2(
    cumul: list, micro: list, macro: list, columns: list, N: int
) -> pd.DataFrame:
    sampled_values = []
    for i in range(N):
        sample = []
        sample_recur2(cumul, micro, macro, columns, sample)
        sampled_values.append(sample)
    return pd.DataFrame(sampled_values, columns=columns)


def sample_recur(
    cumul: list, micro: list, macro: list, columns: list, sample: list
) -> None:
    sampled_value = np.interp(np.random.rand(), cumul[0], micro[0])
    sample.append(sampled_value)

    if len(columns) > 1:
        auxiliar = np.searchsorted(macro[0], sampled_value) - 1
        cumul_aux = []
        micro_aux = []
        macro_aux = []
        for i in range(len(columns) - 1):
            cumul_aux.append(cumul[i + 1][auxiliar])
            micro_aux.append(micro[i + 1][auxiliar])
            macro_aux.append(macro[i + 1][auxiliar])

        sample_recur(cumul_aux, micro_aux, macro_aux, columns[1:], sample)


def sample_df(
    cumul: list, micro: list, macro: list, columns: list, N: int
) -> pd.DataFrame:
    sampled_values = []
    for i in range(N):
        sample = []
        sample_recur(cumul, micro, macro, columns, sample)
        sampled_values.append(sample)
    return pd.DataFrame(sampled_values, columns=columns)


def plot_variables(
    df: pd.DataFrame,
    columns: list,
    nrows: int = 2,
    ncols: int = 3,
    bins: list = None,
    save: bool = False,
    density: bool = True,
    filename: str = "histograms.png",
) -> None:
    """
    Plot histograms of the selected columns of a DataFrame.
    """
    if bins is None:
        bins = [100] * len(columns)
        print(bins)

    if len(columns) > nrows * ncols:
        raise ValueError("The number of columns is greater than the number of subplots")

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    if density:
        # Plot a histogram for each specified column
        for i, col in enumerate(columns):
            df[col].plot(kind="hist", ax=axes[i], title=col, bins=bins[i], density=True)
            axes[i].set_ylabel("Density")
    else:
        # Plot a histogram for each specified column
        for i, col in enumerate(columns):
            df[col].plot(kind="hist", ax=axes[i], title=col, bins=bins[i])
            axes[i].set_ylabel("Frecuency")

    # Remove any empty subplot (since we have 6 plots but only want to use 5) Podria borrar los que no use
    # fig.delaxes(axes[-1])

    plt.tight_layout()  # Adjust layout so plots don't overlap

    if save:
        # Save the figure as a PNG file
        plt.savefig(filename, dpi=300)

    plt.show()


def get_cumulatives_1(
    data: pd.Series, micro_bins: int, macro_bins: int, type: str = "equal_bins"
) -> tuple:
    # cumul = []
    # micro = []
    # macro = []

    if type == "equal_bins":
        counts_micro, micro = np.histogram(data, bins=micro_bins)

        cumul = np.cumsum(counts_micro) / counts_micro.sum()
        cumul = np.insert(cumul, 0, 0)
        # Insert 0 at the beginning to make the frec acum start by
        # 0 and interpolate further with it

        _, macro = np.histogram(data, bins=macro_bins)

        return cumul, micro, macro


def get_cumulatives_2(
    df: pd.DataFrame,
    column_macro1: str,
    column_micro: str,
    macros1: list,
    micro_bins: int,
    macro_bins: int,
    type: str = "equal_bins",
) -> tuple:
    cumul = []
    micro = []
    macro = []

    if type == "equal_bins":
        for i in range(len(macros1) - 1):
            df_filtered = df[
                (df[column_macro1] >= macros1[i])
                & (df[column_macro1] <= macros1[i + 1])
            ]

            cumul_aux, micro_aux, macro_aux = get_cumulatives_1(
                df_filtered[column_micro], micro_bins, macro_bins
            )
            cumul.append(cumul_aux)
            micro.append(micro_aux)
            macro.append(macro_aux)

        return cumul, micro, macro


def get_cumulatives_3(
    df: pd.DataFrame,
    column_macro1: str,
    column_macro2: str,
    column_micro: str,
    macros1: list,
    macros2: list,
    micro_bins: int,
    macro_bins: int,
    type: str = "equal_bins",
) -> tuple:
    cumul = []
    micro = []
    macro = []

    if type == "equal_bins":
        for i in range(len(macros1) - 1):
            df_filtered = df[
                (df[column_macro1] >= macros1[i])
                & (df[column_macro1] <= macros1[i + 1])
            ]
            cumul_aux, micro_aux, macro_aux = get_cumulatives_2(
                df_filtered,
                column_macro2,
                column_micro,
                macros2[i],
                micro_bins,
                macro_bins,
            )
            cumul.append(cumul_aux)
            micro.append(micro_aux)
            macro.append(macro_aux)

        return cumul, micro, macro


def get_cumulatives_4(
    df: pd.DataFrame,
    column_macro1: str,  # First macro column
    column_macro2: str,  # Second macro column
    column_macro3: str,  # Third macro column
    column_micro: str,  # Micro column
    macros1: list,  # List of macro bins edges for the first macro column. dimension: 1
    macros2: list,  # List of macro bins edges for the second macro column. dimension: 2
    macros3: list,  # List of macro bins edges for the third macro column. dimension: 3
    micro_bins: int,  # Number of bins for the cumulative histogram of the micro column
    macro_bins: int,  # Number of bins for the cumulative histogram of the macro4 (useless if this is the last function)
    type: str = "equal_bins",
) -> tuple:
    cumul = []
    micro = []
    macro = []


# def get_cumulatives(
#     cf: pd.DataFrame,
#     columns: list,  #
#     micro_bins: list,  # Number of bins for the cumulative histogram of the micro column
#     macro_bins: list,  # Number of bins for the cumulative histogram of the macro4 (useless if this is the last function)
#     cumul: list,
#     micro: list,
#     macro: list,
# ) -> tuple:

#     number_columns = len(columns)


# def func2(list: list) -> list:
#     if len(list) == 1:
#         auxiliar = [None] * list[0]
#         print(auxiliar)
#         return auxiliar
#     auxiliar = [None] * list[0]
#     for i in range(len(auxiliar)):
#         print(list[1:])
#         auxiliar[i] = func2(list[1:])
#     return auxiliar


# def func1(list: list) -> list:
#     array = [None] * len(list)
#     for i in range(len(list) - 1):
#         array[i + 1] = func2(list[: i + 1])
#     return array


def recur2(
    df: pd.DataFrame,
    columns: list,  #
    micro_bins: list,  # Number of bins for the cumulative histogram of the micro column
    macro_bins: list,  # Number of bins for the cumulative histogram of the macro4 (useless if this is the last function)
    # cumul: list,  #
    # micro: list,  #
    # macros: list,  #
    # number_columns: int,
    type: str = "equal_bins",
) -> tuple:

    if type == "equal_bins":
        # cumul_list = [None] * len(columns)
        # micro_list = [None] * len(columns)
        # macro_list = [None] * len(columns)

        cumul_list = []
        micro_list = []
        macro_list = []

        if len(columns) == 1:
            counts, micro = np.histogram(df[columns[0]], bins=micro_bins[0])
            cumul = np.cumsum(counts)
            total = counts.sum()
            if total:
                cumul = cumul / total
            cumul = np.insert(
                cumul, 0, 0
            )  # Insert 0 at the beginning to make the frec acum start by 0 and interpolate further with it

            cumul_list.append(cumul)
            micro_list.append(micro)
            macro_list.append(None)

            return cumul_list, micro_list, macro_list

        else:
            counts, micro = np.histogram(df[columns[0]], bins=micro_bins[0])
            cumul = np.cumsum(counts)
            total = counts.sum()
            if total:
                cumul = cumul / total
            cumul = np.insert(
                cumul, 0, 0
            )  # Insert 0 at the beginning to make the frec acum start by 0 and interpolate further with it

            _, macro = np.histogram(df[columns[0]], bins=macro_bins[0])

            cumul_list.append(cumul)
            micro_list.append(micro)
            macro_list.append(macro)

            for i in range(len(columns) - 1):
                cumul_list.append([])
                micro_list.append([])
                macro_list.append([])

            for i in range(len(macro) - 1):
                df_filtered = df[
                    (df[columns[0]] >= macro[i]) & (df[columns[0]] <= macro[i + 1])
                ]

                cumul_aux, micro_aux, macro_aux = recur2(
                    df_filtered,
                    columns[1:],
                    micro_bins[1:],
                    macro_bins[1:],
                )

                for j in range(len(cumul_list) - 1):
                    cumul_list[j + 1].append(cumul_aux[j])
                    micro_list[j + 1].append(micro_aux[j])
                    macro_list[j + 1].append(macro_aux[j])

            return cumul_list, micro_list, macro_list

            # for i in range(len(macros) - 1):
            #     df_filtered = df[
            #         (df[columns[0]] >= macros[i]) & (df[columns[0]] <= macros[i + 1])
            #     ]

            #     del df_filtered[columns[0]]

            #     cumul_aux, micro_aux, macro_aux = get_cumulatives(
            #         df_filtered, columns[1:], micro_bins[1:], macro_bins[1:], macros[i]
            #     )

            #     cumul.append(cumul_aux)
            #     micro.append(micro_aux)
            #     macro.append(macro_aux)


def tiempos_ejecucion(
    df: pd.DataFrame,
    columns: list,
    macro_bins: list,
    micro_bins: list,
    N: int,
    save: bool = False,
) -> tuple:

    start_time = time.perf_counter()

    cumul_1, micro_1, macro_1 = get_cumulatives_1(
        df[columns[0]], micro_bins[0], macro_bins[0]
    )

    cumul_2, micro_2, macro_2 = get_cumulatives_2(
        df, columns[0], columns[1], macro_1, micro_bins[1], macro_bins[1]
    )

    cumul_3, micro_3, macro_3 = get_cumulatives_3(
        df,
        columns[0],
        columns[1],
        columns[2],
        macro_1,
        macro_2,
        micro_bins[2],
        macro_bins[2],
    )

    end_time = time.perf_counter()
    time_comulative = end_time - start_time

    start_time = time.perf_counter()

    sampled_values_x, sampled_values_y, sampled_values_u = sample_3d(
        cumul_1, micro_1, macro_1, cumul_2, micro_2, macro_2, cumul_3, micro_3, N
    )

    end_time = time.perf_counter()
    time_sample = end_time - start_time

    start_time = time.perf_counter()

    df_sampled = pd.DataFrame(
        {
            columns[0]: sampled_values_x,
            columns[1]: sampled_values_y,
            columns[2]: sampled_values_u,
        }
    )

    end_time = time.perf_counter()
    time_df = end_time - start_time
    sanitized_columns = [col.replace("/", "d") for col in columns]
    plot_correlated_variables(
        df_sampled,
        columns,
        save=save,
        density=True,
        # fill filename with the order of the columns, the size of macro and micro bins and N
        filename="_".join(sanitized_columns)
        + f"_Macro_bins_{macro_bins}_Micro_bins_{micro_bins}_N_{N}_cumul_{time_comulative}_sample_{time_sample}_df_{time_df}.png",
    )

    return time_comulative, time_sample, time_df


def testeo(
    file: str,
    columns: list,
    micro_bins: list,
    macro_bins: list,
    N: int,
    type: str = "equal_size",
) -> tuple:
    SurfaceSourceFile = kds.SurfaceSourceFile(file, domain={"w": [0, 1]})
    df = SurfaceSourceFile.get_pandas_dataframe()
    del SurfaceSourceFile

    df = df[columns]

    start_time = time.perf_counter()
    cumul, micro, macro = calculate_cumulative_histograms(
        df, columns, micro_bins, macro_bins, type=type
    )
    end_time = time.perf_counter()
    time_cumulative_histograms = end_time - start_time

    start_time = time.perf_counter()
    df_sampled = sample_nd(cumul, micro, macro, columns, N)
    end_time = time.perf_counter()
    time_sampling = end_time - start_time

    kl_divergence = []
    distance = []

    for column in columns:
        kl_divergence.append(
            entropy(np.histogram(df[column])[0], np.histogram(df_sampled[column])[0])
        )
        distance.append(wasserstein_distance(df[column], df_sampled[column]))

    return time_cumulative_histograms, time_sampling, kl_divergence, distance
