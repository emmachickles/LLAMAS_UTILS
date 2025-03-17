import numpy as np
import matplotlib.pyplot as plt
from pypeit.core.wavecal.wvutils import xcorr_shift_stretch, shift_and_stretch
plt.ion()

# Generate a simple example spectrum
x = np.linspace(4000, 7000, 1000)  # Wavelength grid (Angstroms)
y = np.exp(-0.5 * ((x - 5500) / 100)**2)  # Gaussian peak centered at 5500 Å

# Create a shifted and stretched version
true_shift = 5.0  # True shift in pixels
true_stretch = 1.01  # True stretch factor
x_shifted = x * true_stretch + true_shift  # Apply shift and stretch
y_shifted = np.exp(-0.5 * ((x_shifted - 5500) / 100)**2)  # Recalculate the spectrum

# Use xcorr_shift_stretch to recover the shift and stretch
result_out, shift_out, stretch_out, stretch2_out, corr_out, shift_cc, corr_cc = \
    xcorr_shift_stretch(y, y_shifted, shift_mnmx=(-0.1, 0.1))

# Apply the computed shift and stretch to y_shifted
y_corrected = shift_and_stretch(y_shifted, shift_out, stretch_out, stretch2_out)

# Plot the original, shifted, and corrected spectra
plt.figure(figsize=(8, 5))
plt.plot(x, y, label='Original Spectrum (Reference)', linestyle='--', color='black')
plt.plot(x, y_shifted, label='Shifted+Stretched Spectrum', color='red', alpha=0.7)
plt.plot(x, y_corrected, label='Recovered Spectrum (Corrected)', color='blue', linestyle='dotted')
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Intensity')
plt.legend()
plt.title(f'Recovered Shift: {shift_out:.2f} pixels, Stretch: {stretch_out:.4f}')
