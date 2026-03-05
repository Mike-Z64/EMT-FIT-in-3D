import os

# Constants / Global Variables
SPEED_OF_LIGHT = 299792458  # m/s
FREQUENCY = 1.0e9           # 1.0 GHz
WAVELENGTH = SPEED_OF_LIGHT / FREQUENCY

# Working volume dimensions (in meters)
WORKING_VOLUME = {
    'x_min': 0.0,
    'x_max': 3*WAVELENGTH,  # 3 wavelengths in x-direction
    'y_min': 0.0,
    'y_max': 3*WAVELENGTH,
    'z_min': 0.0,
    'z_max': 3*WAVELENGTH
}

def main():
    # clear terminal on Windows
    os.system('cls')
    # Calculating wavelength: λ = c / f
    wavelength = SPEED_OF_LIGHT / FREQUENCY
    
    
    print("--- EMT-FIT-in-3D Project ---")
    print(f"Frequency: {FREQUENCY/1e9} GHz")
    print(f"Speed of Light: {SPEED_OF_LIGHT} m/s")
    print(f"Calculated Wavelength: {wavelength:.4f} meters = {wavelength*1e2:.4f} cm")


if __name__ == "__main__":
    main()