import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import periodogram, correlate
from scipy.cluster.vq import kmeans, vq
from scipy.interpolate import interp1d
import pandas as pd
from sklearn.cluster import KMeans

# 1. Load FITS data
def load_fits(file_path):
    with fits.open(file_path) as hdul:
        data = hdul[1].data  # Adjust HDU index if needed
    return data

# 2. Generate Light Curve by summing counts across all energy channels
def generate_light_curve(data):
    time = data['TIME']
    data_array = data['DataArray']
    counts_array = np.array([list(row) for row in data_array])  # Convert to numpy array
    counts = counts_array.sum(axis=1)
    return time, counts

# 3. Generate Power Density Spectrum
def generate_pds(counts, time_step):
    freqs, power = periodogram(counts, fs=1 / time_step)
    return freqs, power

# 4. Cluster Analysis for Unsupervised Learning using KMeans
def cluster_light_curves(light_curves, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters)
    light_curves = np.array(light_curves, dtype=np.float64).reshape(-1, 1)
    kmeans.fit(light_curves)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return centroids, labels

# 5. Interpolation for Consistent Frequency Bins
def interpolate_power_spectrum(freqs, power, common_freqs):
    interp_func = interp1d(freqs, power, kind='linear', fill_value='extrapolate')
    return interp_func(common_freqs)

# 6. Compute Hardness Intensity (used for Hardness Ratio)
def compute_hardness_intensity(light_curves):
    hardness = [np.std(curve) / np.mean(curve) if np.mean(curve) > 0 else 0 for curve in light_curves]
    return hardness

# 7. Compute Fourier Lag
# def compute_fourier_lag(light_curves):
#     reference_curve = light_curves[0]  # Assuming the first light curve is the reference
#     lags = []
#     for curve in light_curves[1:]:
#         cross_corr = correlate(reference_curve, curve, mode='full')
#         lag = (np.argmax(cross_corr) - len(reference_curve) + 1)
#         lags.append(lag)
#     return np.array(lags) if lags else np.array([0])

# 8. Select Folder Using Tkinter
def select_folder():
    Tk().withdraw()  # Hide the root window
    folder_selected = filedialog.askdirectory(title="Select Folder Containing FITS Files")
    return folder_selected

# 9. Calculate Photon Index (Placeholder for Actual Calculation)
def calculate_photon_index(power_spectrum):
    return np.log10(np.mean(power_spectrum))

# 10. Placeholder for Inner Disc Temperature
def calculate_inner_disc_temperature(power_spectrum):
    return np.mean(power_spectrum) * 0.01

# 11. Placeholder for Quasi Periodic Oscillations (QPO)
def calculate_qpo(power_spectrum):
    return np.max(power_spectrum) - np.min(power_spectrum)

# 12. Placeholder for QPO Amplitude
def calculate_qpo_amplitude(power_spectrum):
    return np.ptp(power_spectrum)

# 13. Hard/Soft Lags Placeholder
def calculate_hard_soft_lags(power_spectrum):
    return np.mean(power_spectrum[:len(power_spectrum)//2]), np.mean(power_spectrum[len(power_spectrum)//2:])

# 14. RMS Amplitude Placeholder
def calculate_rms_amplitude(power_spectrum):
    return np.sqrt(np.mean(power_spectrum ** 2))

# Function to calculate hardness and intensity
def calculate_hardness_and_intensity(time, counts, high_energy_threshold=2, low_energy_threshold=1):
    """
    Calculate the hardness and intensity of the light curve.

    Args:
        time (array): Array of time values from the light curve.
        counts (array): Array of count values from the light curve.
        high_energy_threshold (float): High energy channel threshold.
        low_energy_threshold (float): Low energy channel threshold.

    Returns:
        tuple: (hardness, intensity)
    """
    # Split counts into high and low energy based on thresholds
    high_energy_counts = counts[time >= high_energy_threshold]  # High energy counts
    low_energy_counts = counts[time <= low_energy_threshold]  # Low energy counts

    # Calculate intensity as the total counts (sum of counts)
    intensity = np.sum(counts)

    # Calculate hardness as the ratio of high to low energy counts
    hardness = np.sum(high_energy_counts) / np.sum(low_energy_counts) if np.sum(low_energy_counts) > 0 else 0

    return hardness, intensity

# def compute_fourier_lag_frequency(freqs, power, light_curves):
#     """
#     Compute the Fourier lag for each frequency component in the light curves.

#     Args:
#         freqs (array): Array of frequencies from the power spectrum.
#         power (array): Array of power values corresponding to the frequencies.
#         light_curves (list): List of light curves to compute the lag.

#     Returns:
#         tuple: (frequencies, lags)
#     """
#     if len(light_curves) < 2:
#         raise ValueError("At least two light curves are required to compute the lag.")

#     reference_curve = light_curves[0]  # Assuming the first light curve is the reference
#     lags = []
    
#     # Calculate the lag for each frequency
#     for freq, p in zip(freqs, power):
#         # Cross-correlate the reference curve with each light curve
#         cross_corr = correlate(reference_curve, light_curves[1], mode='full')
#         lag = (np.argmax(cross_corr) - len(reference_curve) + 1)
#         lags.append(lag)
    
#     return np.array(freqs), np.array(lags)


from scipy.fftpack import fft
import numpy as np

def compute_fourier_lag_frequency(freqs, power, light_curves):
    if len(light_curves) < 2:
        raise ValueError("At least two light curves are required to compute the lag.")
    
    reference_curve = light_curves[0]
    target_curve = light_curves[1]
    
    # Perform FFT on both light curves
    fft_ref = fft(reference_curve)
    fft_target = fft(target_curve)
    
    # Compute the cross-spectrum
    cross_spectrum = fft_ref * np.conj(fft_target)
    
    # Calculate the phase difference
    phase_diff = np.angle(cross_spectrum)
    
    # Ensure `freqs` and `phase_diff` have the same length
    min_length = min(len(freqs), len(phase_diff))
    freqs = freqs[:min_length]
    phase_diff = phase_diff[:min_length]
    
    # Replace zeros in `freqs` to avoid division errors
    small_value = 1e-10
    freqs = np.where(freqs == 0, small_value, freqs)
    
    # Sanitize `freqs` and `phase_diff` to remove invalid values
    valid_indices = ~np.isnan(freqs) & ~np.isinf(freqs) & ~np.isnan(phase_diff) & ~np.isinf(phase_diff)
    freqs = freqs[valid_indices]
    phase_diff = phase_diff[valid_indices]
    
    # Handle potential NaN or Inf values after filtering
    freqs = np.nan_to_num(freqs, nan=0.0, posinf=0.0, neginf=0.0)
    phase_diff = np.nan_to_num(phase_diff, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute lags
    lags = phase_diff / (2 * np.pi * freqs)
    
    # Replace invalid lag values with 0
    lags = np.where(np.isfinite(lags), lags, 0)
    
    return freqs, lags




#  18. KMeans Clustering for Accretion States Prediction
def kmeans_predict_accretion_states(input_csv, output_csv, n_clusters=4):
    # Load the data from the CSV
    df = pd.read_csv(input_csv)
    
    # Select relevant features for clustering
    features = ['Photon Index', 'Inner Disc Temperature', 'Hardness Ratio', 'QPO', 'QPO Amplitude', 
                'Hard Lag', 'Soft Lag', 'RMS Amplitude']
    
    # Filter out rows with missing values for the selected features
    df_clean = df[features + ['File']].dropna()
    
    # Initialize and fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(df_clean[features])
    
    # Predict the cluster labels (accretion states)
    df_clean['Accretion State'] = kmeans.labels_
    
    # Map the cluster labels to human-readable accretion states (optional, based on clusters)
    accretion_state_map = {
        0: 'Very High',  # Cluster 0 - You can customize based on analysis
        1: 'Hard',       # Cluster 1 - Customize the mapping
        2: 'Intermediate',  # Cluster 2 - Customize the mapping
        3: 'Soft'        # Cluster 3 - Customize the mapping
    }

    # Apply the mapping to the predicted labels
    df_clean['Accretion State'] = df_clean['Accretion State'].map(accretion_state_map)

    # Save the results (with predicted accretion states) to a new CSV file
    df_clean.to_csv(output_csv, index=False)

    # Print the first few rows of the new CSV
    print(f"Results saved to: {output_csv}")
    print(df_clean.head())
    print(df_clean[['File', 'Accretion State']].head())

def process_fits_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data_files = [f for f in os.listdir(input_folder) if f.endswith('.fits')]

    results = []
    light_curves = []
    combined_power = []
    combined_freqs = []
    accretion_states = []
    hardness_values = []
    intensity_values = []

    for fits_file in data_files:
        fits_path = os.path.join(input_folder, fits_file)
        data = load_fits(fits_path)

        time, counts = generate_light_curve(data)
        light_curves.append(counts)

        time_step = np.mean(np.diff(time))
        freqs, power = generate_pds(counts, time_step)
        combined_power.append(power)
        combined_freqs.append(freqs)

        # Interpolate Power Densities
        min_freq = max(min(freq.min() for freq in combined_freqs), 1e-10)
        max_freq = max(freq.max() for freq in combined_freqs)
        common_freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), 500)

        interpolated_power = [
            interpolate_power_spectrum(freqs, power, common_freqs)
            for freqs, power in zip(combined_freqs, combined_power)
        ]
        mean_power = np.mean(interpolated_power, axis=0)

        # Only compute Fourier lag if there are at least two light curves
        if len(light_curves) >= 2:
            freq, lags = compute_fourier_lag_frequency(freqs, power, light_curves)
            fourier_lag = lags[0] if lags.size > 0 else 0
        else:
            freq, lags = np.array([]), np.array([])
            fourier_lag = 0

        photon_index = calculate_photon_index(mean_power)
        inner_disc_temperature = calculate_inner_disc_temperature(mean_power)
        qpo = calculate_qpo(mean_power)
        qpo_amplitude = calculate_qpo_amplitude(mean_power)
        hardness_intensity = compute_hardness_intensity(light_curves)

        hard_lag, soft_lag = calculate_hard_soft_lags(mean_power)
        rms_amplitude = calculate_rms_amplitude(mean_power)

        # Calculate hardness and intensity
        hardness, intensity = calculate_hardness_and_intensity(time, counts)
        hardness_values.append(hardness)
        intensity_values.append(intensity)

        # Collect results for CSV
        results.append({
            'File': fits_file,
            'Photon Index': photon_index,
            'Inner Disc Temperature': inner_disc_temperature,
            'Hardness Ratio': hardness_intensity[0],
            'QPO': qpo,
            'QPO Amplitude': qpo_amplitude,
            'Hard Lag': hard_lag,
            'Soft Lag': soft_lag,
            'RMS Amplitude': rms_amplitude,
        })

        # Create graphs and save them
        image_folder = os.path.join(output_folder, fits_file.replace('.fits', '_graphs'))
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        plt.figure()
        plt.plot(np.arange(len(counts)), counts, label='Light Curve')
        plt.title(f'Light Curve - {fits_file}')
        plt.xlabel('Time (s)')
        plt.ylabel('Brightness')
        plt.savefig(os.path.join(image_folder, f'{fits_file}_light_curve.png'))
        plt.close()

        plt.figure()
        plt.plot(common_freqs, mean_power, label='Power Spectrum')
        plt.title(f'Power Density Spectrum - {fits_file}')
        plt.xlabel('Frequency (Hz)')
        plt.xlim(0, 0.02)
        plt.ylabel('Power')
        plt.savefig(os.path.join(image_folder, f'{fits_file}_power_density_spectrum.png'))
        plt.close()

        if freq.size > 0 and lags.size > 0:
            plt.figure()
            plt.plot(freq, lags, label='Fourier Lag vs Frequency')
            plt.title(f'Fourier Lag vs Frequency - {fits_file}')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Lag (Samples)')
            plt.legend()
            plt.savefig(os.path.join(image_folder, f'{fits_file}_fourier_lag_frequency.png'))
            plt.close()
            
        # Hardness vs Intensity graph for the current FITS file
        plt.figure(figsize=(8, 6))
        plt.scatter(hardness_values, intensity_values, color='green', label='Hardness vs Intensity')
        plt.title(f'Hardness vs Intensity for {fits_file}')
        plt.xlabel('Hardness')
        plt.ylabel('Intensity')
        plt.legend()
        plt.savefig(os.path.join(image_folder, f'{fits_file}_hardness_intensity.png'))
        plt.close()
        
        # Generate a subplot with all four graphs
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(np.arange(len(counts)), counts, label='Light Curve')
        plt.title(f'Light Curve - {fits_file}')
        plt.xlabel('Time (s)')
        plt.ylabel('Brightness')

        plt.subplot(2, 2, 2)
        plt.plot(common_freqs, mean_power, label='Power Spectrum')
        plt.xlim(0, 0.02)
        plt.title(f'Power Density Spectrum - {fits_file}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')

        # Hardness vs Intensity graph for the current FITS file
        plt.subplot(2, 2, 3)
        plt.scatter(hardness_values, intensity_values, color='green', label='Hardness vs Intensity')
        plt.title(f'Hardness vs Intensity for {fits_file}')
        plt.xlabel('Hardness')
        plt.ylabel('Intensity')
        plt.legend()

        # Fourier Lag vs Frequency Plot
        plt.subplot(2, 2, 4)
        plt.plot(freq, lags, label='Fourier Lag vs Frequency')
        plt.title(f'Fourier Lag vs Frequency - {fits_file}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Lag (Samples)')
        plt.legend()


        plt.tight_layout()
        plt.savefig(os.path.join(image_folder, f'{fits_file}_all_graphs_subplot.png'))
        plt.close()


    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_folder, 'analysis_results.csv'), index=False)

    input_csv = os.path.join(output_folder, 'analysis_results.csv')
    output_csv = os.path.join(output_folder, 'predicted_accretion_states.csv')
    kmeans_predict_accretion_states(input_csv, output_csv)



# 17. Main function to execute the processing pipeline
if __name__ == "__main__":
    input_folder = '/Users/meerthikasr/Documents/Meerthika/100_isro/images'
    output_folder = os.path.join(input_folder, "analysis_results")
    process_fits_images(input_folder, output_folder)
