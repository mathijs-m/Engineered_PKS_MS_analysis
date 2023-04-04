'''
This script extracts extracted ion chromatograms of masses stored in a defined dictionary for each file in a list of files.
The script uses the pyteomics package to achieve this
'''

import numpy as np
from pyteomics import mzxml, auxiliary
from pathlib import Path
#import originpro as op
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
from matplotlib import rc
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

def extract_eic_for_mass(mzxml_object, mass_list, accuracy = 5e-6):
    '''
    This function extracts the EICs for a list of masses from a mzXML object
    :param mzxml_object:   The mzXML object from pyteomics
    :param mass_list:   A list of masses for which the EICs should be extracted
    :param accuracy:    The accuracy in ppm for which the EICs should be extracted
    :return:    A dictionary with the masses as keys and the EICs as values
    '''
    max_id = int(mzxml_object.time[1e3]['num'])
    spectra = [mzxml_object.get_by_id(str(id)) for id in range(1, max_id)]
    eics = dict()
    for mass in mass_list:
        eics[mass] = np.zeros((2,max_id-1), dtype=np.float32)

    # Iterate over the spectra and save the intensity and retention time for each mass
    for spectrum_id, spectrum in enumerate(spectra):
        #print(f'{spectrum_id}/{max_id}\r', end='')
        for peak_id, experimental_mass in enumerate(spectrum['m/z array']):
            for mass in mass_list:
                eics[mass][0, spectrum_id] = spectrum['retentionTime']
                if abs(experimental_mass-mass) < accuracy*experimental_mass:
                    eics[mass][1, spectrum_id] = spectrum['intensity array'][peak_id]
    return eics


def write_origin_file(masses, eics, subfolder):
    '''
    This function writes an Origin file in which the EICs are plotted in a single plot
    :param masses:  A list of masses for which the EICs should be plotted
    :param eics:    A dictionary with the masses as keys and the EICs as values
    :param subfolder:   The subfolder in which the Origin file should be saved
    :return:    None
    '''
    import sys
    def origin_shutdown_exception_hook(exctype, value, traceback):
        '''Ensures Origin gets shut down if an uncaught exception'''
        op.exit()
        sys.__excepthook__(exctype, value, traceback)

    if op and op.oext:
        sys.excepthook = origin_shutdown_exception_hook

    # Set Origin instance visibility.
    if op.oext:
        op.set_show(True)



    origin = op
    origin.new_page()
    for mass_id, mass in enumerate(masses):
        wks = op.new_sheet(type='w', lname=str(mass))
        wks.from_list(0, [t[0] for t in mass], lname='Retention time')
        wks.from_list(1, [t[1] for t in mass], lname='Intensity')
        graph = op.new_graph(lname=str(mass))
        graph_layer_1 = graph[0]
        plot_1 = graph_layer_1.add_plot(wks, coly='B', colx='A', type=200)  # X is col A, Y is col B. 202 is Line + Symbol.
        plot_1.color = '#335eff'
        graph_layer_1.rescale()

        origin.plot(eics[mass][0], eics[mass][1], name='m/z = {}'.format(mass))
        # And also make a nice graph
        origin.graph(name='m/z = {}'.format(mass), x_label='Retention time', y_label='Intensity')
        # Set the the x-ticks to every 2 minutes.
        origin.set_ticks(x_ticks=2 * 60)
        # Set the y-range such that there is a 5% offset between the axes and the data
        origin.set_range(y_range=[1.05 * np.max(eics[mass][1]), 1.05 * np.max(eics[mass][1])])
        # Set the y-ticks to the nearest whole number of that order of magnitude
        origin.set_ticks(y_ticks=10 ** np.floor(np.log10(np.max(eics[mass][1]))))
        # Set the plot size to 8x9 inches
        origin.set_size(size=[8, 9])
        # Set the font size to 12
        origin.set_font(font_size=12)
        # Set the line width to 2
        origin.set_line(line_width=2)
        # Set the line color to navy blue
        origin.set_line(line_color='navy blue')
        # Set the line style to solid
        origin.set_line(line_style='solid')

    # Save the Origin file
    origin.save(subfolder / subfolder.name + '.opj')


def plot(eics, masses, subfolder, time_range, filename, stack_plot=False):
    '''
    This function plots the EICs in a single plot with matplotlib
    :param eics:    A dictionary with the masses as keys and the EICs as values
    :param masses:  A list of masses for which the EICs should be plotted
    :param subfolder:   The subfolder in which the Origin file should be saved
    :param time_range:  The time range in which the EICs should be plotted
    :param stack_plot:  Whether the EICs should be plotted in a stack plot
    :return:    None
    '''

    def format_func(value, tick_number):
        if value == 0:
            return '0'
        return r'${:.1f} \cdot 10^{{{}}}$'.format(value / 10 ** int(np.log10(abs(value))), int(np.log10(abs(value))))

    fig, axs = plt.subplots(len(masses), 1, figsize=(8, 9), squeeze=False, sharex=True)
    for mass_id, mass in enumerate(masses):
        if stack_plot:
            mass_id = 0
        axs[mass_id][0].plot(eics[mass][0], eics[mass][1])

        # Label the axes with time and intensity
        axs[mass_id][0].set_xlabel('Time (min)', fontsize=14)
        axs[mass_id][0].set_ylabel('Intensity', fontsize=14)
        axs[mass_id][0].set_xlim(time_range)
        # Set y-label to scientific notation
        axs[mass_id][0].yaxis.set_major_formatter(plt.FuncFormatter(format_func))
        # axs[mass_id][0].xaxis.get_major_formatter()._usetex = False
        # Set the the x-ticks to every 2 minutes.
        axs[mass_id][0].set_xticks(np.arange(0, np.max(eics[mass][0]), 2))

    # Set the y-range such that there is a 5% offset between the axes and the data


    plt.savefig(filename)
    plt.savefig(filename)
    plt.close()

    # Plot the EICs in a single plot with matplotlib


def parse_mxzml_file(file, masses, stack_plot=False):
    print('Extracting EICs for file {}'.format(file))
    mzxml_object = mzxml.MzXML(str(file.resolve()))
    eics = extract_eic_for_mass(mzxml_object, masses)
    print('...done')
    # Make a subfolder for the data from the current file
    subfolder = file.parent / file.stem

    # Create the subfolder if it does not exist
    if not subfolder.exists():
        subfolder.mkdir()

    # Save the EICs to a text file
    print('Saving EICs to text files')
    for mass in masses:
        np.savetxt(subfolder / f'mz_{mass}.txt', eics[mass].T, delimiter='\t', header='Retention time\tIntensity')

    # Plot the EICs in a single plot with subplots, even if there is only one mass
    print('Plotting EICs')

    def format_func(value, tick_number):
        if value == 0:
            return '0'
        return r'${:.1f} \cdot 10^{{{}}}$'.format(value / 10 ** int(np.log10(abs(value))), int(np.log10(abs(value))))

    #rc('text', usetex=True)
    if stack_plot:
        fig, axs = plt.subplots(len(masses), 1, figsize=(8, 9), squeeze=False, sharex=True)
        for mass_id, mass in enumerate(masses):
            axs[mass_id][0].plot(eics[mass][0], eics[mass][1])
            axs[mass_id][0].set_title(f'{file.stem}: m/z = {mass}')

            # Label the axes with time and intensity
            axs[mass_id][0].set_xlabel('Time (min)', fontsize = 14)
            axs[mass_id][0].set_ylabel('Intensity', fontsize = 14)
            # Set y-label to scientific notation
            axs[mass_id][0].yaxis.set_major_formatter(plt.FuncFormatter(format_func))
            # axs[mass_id][0].xaxis.get_major_formatter()._usetex = False
            # Set the the x-ticks to every 2 minutes.
            axs[mass_id][0].set_xticks(np.arange(0, np.max(eics[mass][0]), 1))

            # Add an annotation with the mass and intensity to the highest peak
            max_I = max(eics[mass][1])
            max_I_index = np.where(eics[mass][1] == max_I)[0][0]
            max_time = eics[mass][0][max_I_index]
            axs[0][0].annotate(f'{max_time:.2f} min; m/z ({max_I:.2e})', xy=(max_time, max_I), xytext=(max_time, max_I * 1.1),
                                        arrowprops=dict(facecolor='black', shrink=0.05))
            plt.savefig(subfolder / f'mz_{mass}.png')
            plt.savefig(subfolder / f'mz_{mass}.svg')
            plt.close()
    elif not stack_plot:
        for mass_id, mass in enumerate(masses):
            fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False, sharex=True)
            axs[0][0].plot(eics[mass][0], eics[mass][1])
            axs[0][0].set_title(f'{file.stem}: m/z = {mass}')

            # Label the axes with time and intensity
            axs[0][0].set_xlabel('Time (min)', fontsize = 14)
            axs[0][0].set_ylabel('Intensity', fontsize = 14)
            # Set y-label to scientific notation
            axs[0][0].yaxis.set_major_formatter(plt.FuncFormatter(format_func))
            #axs[mass_id][0].xaxis.get_major_formatter()._usetex = False
            # Set the the major x-ticks to every 4 minutes.
            axs[0][0].set_xticks(np.arange(0, np.max(eics[mass][0]), 4))
            # Set the minor x-ticks to every 2 minutes.
            axs[0][0].set_xticks(np.arange(0, np.max(eics[mass][0]), 2), minor=True)
            # Add an annotation with the mass and intensity to the highest peak
            max_I = max(eics[mass][1])
            max_I_index = np.where(eics[mass][1] == max_I)[0][0]
            max_time = eics[mass][0][max_I_index]
            axs[0][0].annotate(f'{max_time:.2f} min; m/z ({max_I:.2e})', xy=(max_time, max_I), xytext=(max_time*0.9, max_I * 1.1),
                                        arrowprops=dict(facecolor='black', shrink=0.05))
            plt.savefig(subfolder / f'mz_{mass}.png')
            plt.savefig(subfolder / f'mz_{mass}.svg')
            plt.close()


    # Write an Origin file using the originpro package
    #print('Writing Origin file')
    #write_origin_file(masses, eics, subfolder)


def main():
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Extract EICs from mzXML files, plot them in png, svg and Origin, and save them to text files')
    parser.add_argument('-f', '--folder', help='The folder in which the mzXML files are stored')
    parser.add_argument('-m', '--masses', help='A comma-separated list of the masses for which the EICs should be extracted')
    parser.add_argument('-a', '--accuracy', help='The accuracy in ppm for which the EICs should be extracted', default=5e-6)

    args = parser.parse_args()

    # Define the folder in which the mzXML files are stored
    folder = Path(args.folder)

    # Define the masses for which the EICs should be extracted
    if args.masses is None:
        masses = [445.1624, 462.1889, 459.178, 476.2045, 363.1205, 380.147, 487.173, 504.1995, 501.1886,
                  518.2151, 405.1311, 422.1576, 469.1624, 486.1889, 483.178, 500.2045, 387.1205, 404.147,
                  529.1835, 546.21, 543.1992, 560.2257, 447.1417, 464.1682, 511.173, 528.1995, 525.1886,
                  542.2151, 429.1311, 446.1576, 531.1992, 548.2257, 545.2148, 562.2413, 449.1573, 466.1838,
                  513.1886, 530.2151, 527.2042, 544.2308, 431.1467, 448.1732, 513.1886, 530.2151, 527.2043,
                  544.2308, 431.1467, 448.1732, 495.178, 512.2046, 509.1937, 526.2202, 413.1362, 430.1627]
    else:

        masses = [float(mass) for mass in args.masses.split(',')]

    # Get the list of mzXML files
    files = get_mzxml_files(folder)

    # Iterate over the files and extract the EICs
    with mp.Pool(max(1, mp.cpu_count() - 2)) as pool:
        pool.starmap(parse_mxzml_file, [(file, masses) for file in files])

if __name__=='__main__':
    main()