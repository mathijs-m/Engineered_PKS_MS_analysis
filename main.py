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
import pickle as pkl
import os

def get_mzxml_files(folder):
    '''
    This function returns a list of mzXML files in a folder
    '''
    files = []
    for file in folder.iterdir():
        if file.suffix == '.mzXML':
            files.append(file)
    return files


def check_isotope_pattern(spectrum, ref_peak_index, delta_mass, isotope_intensity_range, accuracy):
    '''
    This function checks whether the isotope pattern of a peak is present in a spectrum
    :param spectrum:    The spectrum in which the isotope pattern should be checked
    :param ref_peak_index:  The index of the reference peak in the spectrum
    :param delta_mass:  The mass difference between isotopes
    :param isotope_intensity_range: The intensity of the isotopes. Here the intensity of Cl37 and Cl39. The list gives
                the range of intensities of the second isotope (Cl37) to the first isotope (Cl35) that are accepted
    :param accuracy:    The accuracy in ppm for which the EICs should be extracted
    :return:    True if the isotope pattern is present, False otherwise
    '''

    parent_intensity = spectrum['intensity array'][ref_peak_index]
    parent_mass = spectrum['m/z array'][ref_peak_index]
    for delta_peak_id, isotope_mass in enumerate(spectrum['m/z array'][ref_peak_index:]):
        if isotope_mass > parent_mass+delta_mass*1.1:
            break
        relative_isotope_intensity = spectrum['intensity array'][delta_peak_id + ref_peak_index]/parent_intensity
        if abs(isotope_mass-(parent_mass+delta_mass)) < accuracy*parent_mass:
            if isotope_intensity_range[0] < relative_isotope_intensity < isotope_intensity_range[1]:
                return True


def extract_eic_for_mass(mzxml_object, mass_list, accuracy = 10e-6, cutoff=1e3, delta_mass=1.99705,
                         isotope_intensity_range=[0.2, 2]):
    '''
    This function extracts the EICs for a list of masses from a mzXML object
    :param mzxml_object:   The mzXML object from pyteomics
    :param mass_list:   A list of masses for which the EICs should be extracted
    :param accuracy:    The accuracy in ppm for which the EICs should be extracted
    :param cutoff:  The minimum intensity for which the EICs should be extracted
    :param delta_mass:  The mass difference between isotopes. Here the mass difference between Cl35 and Cl37
    :param isotope_intensity:    The intensity of the isotopes. Here the intensity of Cl37 and Cl39. The list gives
                the range of intensities of the second isotope (Cl37) to the first isotope (Cl35) that are accepted
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
                    # No check for isotope pattern for bacillaene
                        eics[mass][1, spectrum_id] = intensity
    return eics


def format_func(value, pos=None):
    '''
    This function formats the y-axis of the EICs to scientific notation in nice LaTeX
    :param value:   The value to be formatted
    :param tick_number: The tick number
    :return:    The formatted value
    '''
    if value == 0:
        return '0'
    return r'${:.1f} \cdot 10^{{{}}}$'.format(value / 10 ** int(np.log10(abs(value))), int(np.log10(abs(value))))


def plot_compound(sample_data, file_root, title_separator, stack_plot=False, set_title=False, normalize = True,
                  row_length=5):
    '''
    This function plots the EICs for a compound
    :param sample_data: The dictionary of dataframes with data for the sample
    :param file_root:   Pathlib object with the root of the file name
    :param title_separator: The separator between the compound name and the retention time
    :param stack_plot:  Whether the EICs should be stacked or not
    :param set_title:   Whether the title should be set or not
    :param normalize:   Whether the EICs should be normalized or not
    :param row_length:  The number of EICs per row
    :return:    None
    '''
    compounds = list(sample_data.keys())
    time_range = [2, 9]
    fig_size = (36, 18)

    if stack_plot:
        num_rows = min(len(sample_data), row_length)
        num_cols = (len(sample_data)-1)//row_length+1
        fig, axs = plt.subplots(num_rows, num_cols, figsize=fig_size, squeeze=False, sharex=True)
        colors = cmocean.cm.haline(
            np.linspace(0, 0.9, max([len(sample_data[compound].columns) for compound in compounds])))
    else:
        num_rows = 1
        num_cols = 1
        fig, axs = plt.subplots(1, 1, figsize=fig_size, squeeze=False, sharex=True)
        colors = cmocean.cm.haline(
            np.linspace(0, 0.9, np.sum([len(sample_data[compound].columns) for compound in compounds])))


    if set_title:
        title = title_separator.join(file_root.name.split(title_separator)[1:])
        title = title.replace('_', '-')
        fig.suptitle(title, fontsize=16)

    for compound_id, compound in enumerate(sample_data):

        if stack_plot:
            plot_num = compound_id%row_length
            row_num = compound_id//row_length
        else:
            plot_num = 0
            row_num = 0

        for isomer_id, isomer in enumerate(sample_data[compound]):
            # Skip the retention time column
            if isomer == 'Retention time':
                continue

            ## Plot the EICs. If stack_plot is True, offset each EIC by 0.01
            # If stackplot, offset each EIC by 0.01
            if stack_plot:
                if normalize:
                    offset = 0.02*isomer_id
                else:
                    offset = 0
            else:
                offset = 0

            # Check if the EIC is empty, if so, don't normalize it
            if np.sum(sample_data[compound][isomer]) == 0 or not normalize:
                axs[plot_num][row_num].plot(sample_data[compound]['Retention time'],
                                            sample_data[compound][isomer]+offset, color=colors[isomer_id-1],
                                            label=isomer)
            else:
                axs[plot_num][row_num].plot(sample_data[compound]['Retention time'],
                                            sample_data[compound][isomer]/np.max(sample_data[compound][isomer])+offset,
                                            color=colors[isomer_id-1], label=isomer)
            # Set the line-width to 0.75
            axs[plot_num][row_num].lines[-1].set_linewidth(0.75)

        # Label the axes with time and intensity. Also only label the x-axis of the bottom plot
        if compound_id%row_length==(row_length-1):
            axs[plot_num][row_num].set_xlabel('Time (min)', fontsize=10)
            axs[plot_num][row_num].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
            axs[plot_num][row_num].xaxis.set_major_formatter(FormatStrFormatter('%g'))
            # Set the x-ticks to every 2 minutes.
            axs[plot_num][row_num].set_xticks(np.arange(0, np.max(sample_data[compound]['Retention time']), 1))
        else:
            axs[plot_num][row_num].set_xlabel('')  # Set an empty x-axis label
            axs[plot_num][row_num].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        # Set the x-range to be between 4 and 16 minutes
        axs[plot_num][row_num].set_xlim(time_range[0], time_range[1])

        ## Set the axis labels
        if normalize:
            # Set the y-ticks to be between 0 and 1
            axs[plot_num][row_num].set_yticks(np.arange(0, 1.1, 0.5))
            # Set the y-range to be between 0 and 1.1
            axs[plot_num][row_num].set_ylim(0, 1.2)
        else:
            # Set the y-ticks to the maximum intensity of the EICs
            max_intensity = np.max([np.max(sample_data[compound][isomer]) for isomer in sample_data[compound]])
            divider = 5 if np.log10(max_intensity)%1 > np.log10(2) else 10
            axs[plot_num][row_num].set_yticks(np.arange(0, max_intensity, 10**np.ceil(np.log10(max_intensity))/divider))
            axs[plot_num][row_num].yaxis.set_major_formatter(format_func)
            # Set the x-grid to be on
            axs[plot_num][row_num].grid(axis='x', color='gainsboro', linestyle='-', linewidth=0.5)


        # Label the middle y-axis of the stacked plot with 'Normalized intensity'
        y_axis_label = 'Normalized intensity' if normalize else 'Intensity'
        if stack_plot:
            if len(sample_data) == 1 or compound_id%row_length/(min(row_length, len(sample_data))//2) == 1:
                axs[plot_num][row_num].set_ylabel(y_axis_label, fontsize=10)
        # Label the left y-axis of the non-stacked plot with 'Normalized intensity'
        else:
            axs[plot_num][row_num].set_ylabel(y_axis_label, fontsize=10)

        # Set the title of the compound in the upper left corner of the graph and make it bold
        axs[plot_num][row_num].set_title(' '+compound, fontsize=10, loc='left', pad=-10, fontweight='bold')
        # Make a legend string that also mentions the maximum intensity of the EIC
        legend_strings = [f'{isomer:.4f} - {format_func(np.max(sample_data[compound][isomer]))}'
                          for isomer in sample_data[compound] if isomer != 'Retention time']
        # Set the legend on the right side of the graph and have it constrained to the heigth of the graph
        legend = axs[plot_num][row_num].legend(legend_strings,loc='center left', bbox_to_anchor=(1, 0.5),
                                                             ncol=1, frameon=False, fontsize=8)
        '''
        # Set the legend title to 'Isomer - Max intensity' and align it to the left for the top legend
        if compound_id == 0:
            legend.set_title(f'{"m/z":12}   Max intensity')
            legend.get_title().set_x(-50)
        '''
        # Set the legend title to be in the same font size as the rest of the legend
        plt.setp(legend.get_title(), fontsize=8)

        fig.tight_layout()
        #fig.set_size_inches((num_rows * 2, num_cols * 7)) with num_rows = 5 and num_cols = 2 --> 10, 14
        fig.set_size_inches((num_cols * 5, num_rows * 7/3))

    # Save the figure

    plt.savefig(str(file_root) + '_unnormalized.svg', dpi=300, bbox_inches='tight')
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


def save_to_txt(sample_data, file_root):
    '''
    This function saves the EICs to a single text file, with each compound separated by the compound name
    :param sample_data: The dictionary of dataframes with data for the sample
    :param file_root:   The file root name of the output Excel file
    :return:    None
    '''
    # Make a folder for the text files
    if not os.path.exists(str(file_root)):
        os.makedirs(str(file_root))

    # Write each dataframe to a different text file.
    # Each text file is named after the compound name
    # and contains the EIC data for that compound
    for compound in sample_data:
        with open(str(file_root) + '/' + compound + '.txt', 'w') as f:
            # write to csv file
            sample_data[compound].to_csv(f, index=False, sep='\t')

    # Additionally write all the data to a single text file
    with open(str(file_root) + '.txt', 'w') as f:
        for compound in sample_data:
            f.write('"' + compound + '"\n')
            f.write(sample_data[compound].to_string(index=False))
            f.write('\n\n')



def parse_mzxml_file(file, masses, accuracy, cutoff, stack_plot=False, use_pickle=True,
                     show_title=True, normalize=True, title_separator='-'):
    print('Extracting EICs for file {}'.format(file))
    if use_pickle:
        try:
            with open(str(file.parent / file.stem) + '.pkl', 'rb') as f:
                eics = pkl.load(f)
            print('...done')
        except FileNotFoundError:
            mzxml_object = mzxml.MzXML(str(file.resolve()))
            eics = extract_eic_for_mass(mzxml_object, masses, accuracy, cutoff)
            with open(str(file.parent / file.stem) + '.pkl', 'wb') as f:
                pkl.dump(eics, f)
    else:
        mzxml_object = mzxml.MzXML(str(file.resolve()))
        eics = extract_eic_for_mass(mzxml_object, masses, accuracy, cutoff)
    print('...done')
    # Obtain the file root
    file_root = file.parent / file.stem

    # Save the EICs to a text file
    sample_data = dict()
    for compound in masses:
        compound_data = pd.DataFrame()
        compound_data['Retention time'] = eics[masses[compound][0][0]][0]
        for isomer in masses[compound]:
            for adduct_mass in isomer:
                compound_data[adduct_mass] = eics[adduct_mass][1]
        sample_data[compound] = compound_data

    print('Saving EICs to text files')
    # Save the EICs to an Excel file
    #save_to_excel(sample_data, file_root)
    save_to_txt(sample_data, file_root)

    plot_compound(sample_data, file_root, title_separator, stack_plot, show_title, normalize)


def main():
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Extract EICs from mzXML files, plot them in png, svg and Origin, and '
                                                 'save them to text files')
    parser.add_argument('-f', '--folder', help='The folder in which the mzXML files are stored')
    parser.add_argument('-a', '--accuracy', help='The accuracy in ppm for which the EICs should be extracted',
                        default=5e-6, type=float)
    parser.add_argument('-c', '--cutoff', help='The minimum intensity for which the EICs should be extracted',
                        default=1e3, type=float)
    parser.add_argument('-p', '--pickle', help='Use pickle to store the EICs', action='store_true', default=False)
    parser.add_argument('-t', '--title', help='Show the sample title in the plot', action='store_true', default=False)
    parser.add_argument('-st', '--stack', help='Plot the EICs in a stacked plot', action='store_false', default=True)
    parser.add_argument('-n', '--normalize', help='Normalize the EICs', action='store_true', default=False)
    parser.add_argument('-ts', '--title-separator', help='Separator to remove title prefix, e.g., experiment code.', type=str, default='-')

    args = parser.parse_args()

    # Define the folder in which the mzXML files are stored
    folder = Path(args.folder)

    # Define the masses for which the EICs should be extracted
    mass_H = 1.007825032
    mass_Na = 22.989218
    masses = {}
    '''
    masses = {'Intermediate ACP2': [[190.1074]],
              'Intermediate ACP3-4': [[218.1387]],
              'Intermediate ACP5': [[262.1649]],
              'Intermediate ACP6': [[244.1543]],
              'Keto extension intermediate PksL KS4': [[286.1649]],
              'Reduction intermediate PksL KS4': [[288.1805]],
              'Dehydration intermediate PksL KS4': [[270.1700]],
              'Double extension intermediate PksL KS4': [[328.1755]],
              'Dehydrated double extension intermediate PksL KS4': [[310.1649]],
              'Bacillibactin': [[883.2628]]
              }
    '''
    masses.update({'Intermediate ACP10': [[310.2013]],
              'Intermediate ACP11': [[336.2169]],
              'Keto extension intermediate PksM8': [[378.2275]],
              'Reduction intermediate PksM KS8': [[380.2431]],
              'Dehydration intermediate PksM KS8': [[362.2326]],
              'Double extension intermediate PksM KS8': [[420.2381]],
              'Dehydrated double extension intermediate PksM KS8': [[402.2275]],
              'Bacillibactin': [[883.2628]]
              })
    '''
    masses.update({'Dehydrated reduced double extension intermediate PksL KS4': [[312.1805]],
              'Double extension and arginylation intermediate PksL KS4': [[484.2766]],
              'Double extension, reduction and arginylation intermediate PksL KS4': [[468.2817]],
              'Double extension, dehydration and arginylation intermediate PksL KS4': [[452.2867]],
              'Bacillibactin': [[883.2628]],
              })
    '''

    for compound in masses:
        mass = masses[compound][0][0]
        masses[compound][0] = [mass, mass - mass_H + mass_Na]

    # Get the list of mzXML files
    files = get_mzxml_files(folder)

    # Iterate over the files and extract the EICs
    with mp.Pool(max(1, mp.cpu_count() - 2)) as pool:
        pool.starmap(parse_mzxml_file, [(file, masses, args.accuracy, args.cutoff, args.stack, args.pickle, args.title,
                                         args.normalize, args.title_separator)
                                        for file in files])

if __name__=='__main__':
    main()