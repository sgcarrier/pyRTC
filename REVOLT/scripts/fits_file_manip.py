from astropy.io import fits
import matplotlib.pyplot as plt

def extra_data_from_fits(fits_file_path):

    try:
        # Open the FITS file
        with fits.open(fits_file_path) as hdul:
            # Access the primary HDU (usually contains the image data)
            primary_hdu = hdul[0]

            # Get the image data and header
            image_data = primary_hdu.data
            header_info = primary_hdu.header

        return image_data, header_info

    except FileNotFoundError:
        print(f"Error: The file '{fits_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")