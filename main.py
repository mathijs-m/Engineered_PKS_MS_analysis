'''
This script extracts extracted ion chromatograms of masses stored in a defined dictionary for each file in a list of files.
The script uses the pyteomics package to achieve this
'''

import numpy as np
import pandas as pd
from pyteomics import mzxml, auxiliary
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
import cmocean
import argparse
import multiprocessing as mp

def get_mzxml_files(folder):
    '''
    This function returns a list of mzXML files in a folder
    '''
    files = []
    for file in folder.iterdir():
        if file.suffix == '.mzXML':
            files.append(file)
    return files

def extract_eic_for_mass(mzxml_object, mass_list, accuracy = 10e-6, cutoff=1e3):
    '''
    This function extracts the EICs for a list of masses from a mzXML object
    :param mzxml_object:   The mzXML object from pyteomics
    :param mass_list:   A list of masses for which the EICs should be extracted
    :param accuracy:    The accuracy in ppm for which the EICs should be extracted
    :param cutoff:  The minimum intensity for which the EICs should be extracted
    :return:    A dictionary with the masses as keys and the EICs as values
    '''
    max_id = int(mzxml_object.time[1e3]['num'])
    spectra = [mzxml_object.get_by_id(str(id)) for id in range(1, max_id)]
    eics = dict()
    mass_list = set(adduct_mass for compound in mass_list for isomer in mass_list[compound] for adduct_mass in isomer)
    for mass in mass_list:
        eics[mass] = np.zeros((2,max_id-1), dtype=np.float32)

    # Iterate over the spectra and save the intensity and retention time for each mass
    for spectrum_id, spectrum in enumerate(spectra):
        #print(f'{spectrum_id}/{max_id}\r', end='')
        for peak_id, experimental_mass in enumerate(spectrum['m/z array']):
            intensity = spectrum['intensity array'][peak_id]
            for mass in mass_list:
                eics[mass][0, spectrum_id] = spectrum['retentionTime']
                if intensity > cutoff and abs(experimental_mass-mass) < accuracy*experimental_mass:
                    eics[mass][1, spectrum_id] = intensity
    return eics


def format_func(value):
    '''
    This function formats the y-axis of the EICs to scientific notation in nice LaTeX
    :param value:   The value to be formatted
    :param tick_number: The tick number
    :return:    The formatted value
    '''
    if value == 0:
        return '0'
    return r'${:.1f} \cdot 10^{{{}}}$'.format(value / 10 ** int(np.log10(abs(value))), int(np.log10(abs(value))))


def plot_compound(sample_data, file_root, stack_plot=False):
    '''
    This function plots the EICs for a compound
    :param sample_data: The dictionary of dataframes with data for the sample
    :param subfolder:   The subfolder in which the plots should be saved
    :param stack_plot:  Whether the EICs should be stacked or not
    :return:    None
    '''
    compounds = list(sample_data.keys())
    colors = cmocean.cm.haline(np.linspace(0, 0.9, len(sample_data[compounds[0]])))

    fig, axs = plt.subplots(len(sample_data), 1, figsize=(16, 18), squeeze=False, sharex=True)
    for compound_id, compound in enumerate(sample_data):
        if stack_plot:
            plot_num = compound_id
        else:
            plot_num = 0
        for isomer_id, isomer in enumerate(sample_data[compound]):
            if isomer == 'Retention time':
                continue
            # Plot the normalized EICs
            # If stackplot, offset each EIC by 0.01
            if stack_plot:
                offset = 0.02*isomer_id
            else:
                offset = 0

            # Check if the EIC is empty, if so, don't normalize it
            if np.sum(sample_data[compound][isomer]) == 0:
                axs[plot_num][0].plot(sample_data[compound]['Retention time'], sample_data[compound][isomer]+offset, color=colors[isomer_id-1], label=isomer)
            else:
                axs[plot_num][0].plot(sample_data[compound]['Retention time'], sample_data[compound][isomer]/np.max(sample_data[compound][isomer])+offset, color=colors[isomer_id-1], label=isomer)
            # Set the line-width to 0.75
            axs[plot_num][0].lines[-1].set_linewidth(0.75)

        # Label the axes with time and intensity. Also only label the x-axis of the bottom plot
        if compound_id == len(sample_data)-1:
            axs[plot_num][0].set_xlabel('Time (min)', fontsize=10)
            axs[plot_num][0].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
            axs[plot_num][0].xaxis.set_major_formatter(FormatStrFormatter('%g'))
            # Set the x-ticks to every 2 minutes.
            axs[plot_num][0].set_xticks(np.arange(0, np.max(sample_data[compound]['Retention time']), 0.5))
        else:
            axs[plot_num][0].xaxis.set_ticklabels([])

        # Set the x-range to be between 4 and 8 minutes
        axs[plot_num][0].set_xlim(4, 8)
        # Set the y-ticks to be between 0 and 1
        axs[plot_num][0].set_yticks(np.arange(0, 1.1, 0.5))
        # Set the y-range to be between 0 and 1.1
        axs[plot_num][0].set_ylim(0, 1.2)

        # Label the middle y-axis of the stacked plot with 'Normalized intensity'
        if stack_plot:
            if compound_id%(np.floor(len(sample_data)/2)) == 0 and compound_id not in [0, len(sample_data)-1]:
                axs[plot_num][0].set_ylabel('Normalized intensity', fontsize=10)
        # Label the left y-axis of the non-stacked plot with 'Normalized intensity'
        else:
            axs[plot_num][0].set_ylabel('Normalized intensity', fontsize=10)

        # Set the title of the compound in the upper left corner of the graph
        axs[plot_num][0].set_title(' '+compound, fontsize=10, loc='left', pad=-10)
        # Make a legend string that also mentions the maximum intensity of the EIC
        legend_strings = [f'{isomer:.4f} - {format_func(np.max(sample_data[compound][isomer]))}'
                          for isomer in sample_data[compound] if isomer != 'Retention time']
        # Set the legend on the right side of the graph and have it constrained to the heigth of the graph
        legend = axs[compound_id][0].legend(legend_strings,loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, frameon=False, fontsize=8)
        '''
        # Set the legend title to 'Isomer - Max intensity' and align it to the left for the top legend
        if compound_id == 0:
            legend.set_title(f'{"m/z":12}   Max intensity')
            legend.get_title().set_x(-50)
        '''
        # Set the legend title to be in the same font size as the rest of the legend
        plt.setp(legend.get_title(), fontsize=8)

        fig.tight_layout()
        fig.set_size_inches(8, fig.bbox.height / fig.dpi)

    # Save the figure
    plt.savefig(str(file_root) + '.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_to_excel(sample_data, file_root):
    '''
    This function saves the EICs to an excel file
    :param sample_data: The dictionary of dataframes with data for the sample
    :param file_root:   The file root name of the output Excel file
    :return:    None
    '''
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    with pd.ExcelWriter(str(file_root) + '.xlsx') as writer:
        # Write each dataframe to a different worksheet.
        for compound in sample_data:
            sample_data[compound].to_excel(writer, sheet_name=compound, index=False)


def parse_mzxml_file(file, masses, accuracy, cutoff, stack_plot=False):
    print('Extracting EICs for file {}'.format(file))
    mzxml_object = mzxml.MzXML(str(file.resolve()))
    eics = extract_eic_for_mass(mzxml_object, masses, accuracy, cutoff)
    print('...done')
    # Obtain the file root
    file_root = file.parent / file.stem


    # Save the EICs to a text file
    print('Saving EICs to text files')
    sample_data = dict()
    for compound in masses:
        compound_data = pd.DataFrame()
        compound_data['Retention time'] = eics[masses[compound][0][0]][0]
        for isomer in masses[compound]:
            for adduct_mass in isomer:
                compound_data[adduct_mass] = eics[adduct_mass][1]
        sample_data[compound] = compound_data

    # Save the EICs to an Excel file
    save_to_excel(sample_data, file_root)
    plot_compound(sample_data, file_root, stack_plot)


def main():
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Extract EICs from mzXML files, plot them in png, svg and Origin, and save them to text files')
    parser.add_argument('-f', '--folder', help='The folder in which the mzXML files are stored')
    parser.add_argument('-a', '--accuracy', help='The accuracy in ppm for which the EICs should be extracted', default=5e-6, type=float)
    parser.add_argument('-c', '--cutoff', help='The minimum intensity for which the EICs should be extracted',
                        default=1e3, type=float)

    args = parser.parse_args()

    # Define the folder in which the mzXML files are stored
    folder = Path(args.folder)

    # Define the masses for which the EICs should be extracted
    masses = {'1': [[445.1624, 462.1889], [459.178, 476.2045], [363.1205, 380.147]],
              '2lin': [[487.173, 504.1995], [501.1886, 518.2151], [405.1311, 422.1576]],
              '2cyc': [[469.1624, 486.1889], [483.178, 500.2045], [387.1205, 404.147]],
              '3lin': [[529.1835, 546.21], [543.1992, 560.2257], [447.1417, 464.1682]],
              '3cyc': [[511.173, 528.1995], [525.1886, 542.2151], [429.1311, 446.1576]],
              '4lin': [[531.1992, 548.2257], [545.2148, 562.2413], [449.1573, 466.1838]],
              '4cyc': [[513.1886, 530.2151], [527.2042, 544.2308], [431.1467, 448.1732]],
              '5lin': [[513.1886, 530.2151], [527.2043, 544.2308], [431.1467, 448.1732]],
              '5cyc': [[495.178, 512.2046], [509.1937, 526.2202], [413.1362, 430.1627]]}

    # Get the list of mzXML files
    files = get_mzxml_files(folder)

    # Iterate over the files and extract the EICs
    with mp.Pool(max(1, mp.cpu_count() - 2)) as pool:
        pool.starmap(parse_mzxml_file, [(file, masses, args.accuracy, args.cutoff, True) for file in files])

if __name__=='__main__':
    main()