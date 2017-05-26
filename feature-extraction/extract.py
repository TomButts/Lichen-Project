'''
The feature extraction tool

This tool runs from the command line and allows features to be extracted
from an ordered image directory and a corresponding lables file.

The tool uses configuration information in order to determine which extractor
to use and what the input and output details are.

For regular extraction following config details use '-e'

For an intermitent sleep mode that schedules breaks to prevent
CPU overheating and perfromance drops use '-s'
'''

import sys
import os
import imp
import warnings
import getopt
from tools.convert_dataset import convert_dataset

def warn(*args, **kwargs):
    pass

def usage():
    print(__doc__)

if __name__ == "__main__":
    warnings.warn = warn

    try:
        opts, remainder = getopt.getopt(sys.argv[1:], 'e:s:u:')

        for opt, arg in opts:
            if opt in ('-e'):
                config_base_dir = os.path.abspath('configs/')
                path = config_base_dir + '/' + arg + '.py'

                # 'arg' is the name of the config file without the file ext
                config = imp.load_source(arg, path)

                # Regulare feature extraction
                convert_dataset(config.options)

                exit()
            elif opt in ('-s'):
                config_base_dir = os.path.abspath('configs/')
                path = config_base_dir + '/' + arg + '.py'

                # 'arg' is the name of the config file without the file ext
                config = imp.load_source(arg, path)

                # intermitent sleep mode
                convert_dataset(config.options, 's')

                exit()
            elif opt in ('-u'):
                # usage()
                exit()
    except getopt.GetoptError as err:
        # print(err)
        usage()
        exit()
