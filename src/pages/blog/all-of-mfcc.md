---
title: All of MFCC
date: 2020-09-07
thumb_image: images/sound.jpeg
image: images/sound.jpeg
excerpt: Sound is a wave. To store sound in a computer, waves of air pressure are
  converted into voltage via microphone. It is then sampled with an analog-to-digital
  converter where the output is stored as one dimensional array of number.
template: post

---
Sound is a wave. To store sound in a computer, waves of air pressure are converted into voltage via microphone. It is then sampled with an analog-to-digital converter where the output is stored as one dimensional array of number. Simply saying, audio file is just an array of amplitudes sampled with certain rate known as sampling rate. As a meta-property an audio file has sample rate, number of channels of sound and precision (bit depth).

A standard telephone audio has sampling rate of 8 kHz and 16-bit precision.

You might have heard a familiar term bit-rate (bit per second) which is sometimes used to measure the overall quality of an audio.  
bit-rate = sample rate * precision * no. of channels

A raw audio signal is high-dimensional and difficult to model as there are many information but consisting of many unwanted signals especially in case of human speech sound.

**Mel-frequency Cepstrum** **Coefficients** (**MFCC)** tries to model the audio in a format where it perform those type of filtering that correlates to human auditory system and their low dimensionality. It is the most commonly used features for Automatic Speech Recognition (ASR). **Mel-frequency Cepstrum** (**MFC**) is a representation of the short-term power-spectrum of a sound, based on a linear cosine transform of a [log power spectrum](https://en.wikipedia.org/wiki/Power_spectrum "Power spectrum") on a nonlinear [Mel-scale](https://en.wikipedia.org/wiki/Mel_scale "Mel scale") of frequency and MFCC are the coefficients that collectively make up the MFC.

Following are the steps to compute MFCC.

I will use a example sound wave and process it in python to make it more clear.

```python
  import librosa 
  import numpy as np 
  import librosa.display
  import matplotlib.pyplot as plt
  import sounddevice as sd
  y, sr = librosa.load("00a80b4c8a.flac", sr=16000)
```

![initial signal](/images/y.png "Initial Signal")

1. **Pre-emphasis**

   This is the first step in feature generation. In speech production, high frequencies usually have smaller magnitudes compared to lower frequencies. So in order to counter the effect we apply pre-emphasis signal to amply the amplitude of high frequencies and lower the amplitude of lower frequencies.

   If $ x(t) $ is the signal,

   $$
   y(t) = x(t) - \\alpha x(t-1)
   $$

   where, $\\alpha$ is generally 0.95 or 0.97.

   ```python
       alpha = 0.97
       y_emp = np.append(y[0], y[1:] - alpha * y[:-1])
   ```

   Here's the visualization of pre-emphasized signal.

   ![amplified signal](/images/yemp.png "Pre-emphasized signal")
2. **Framing**

   Acoustic signal is perpetually changing in speech. But studies show that the characteristics of voice signal remains fixed in a short interval of time (called quasi-stationary signals). So while modelling the signal we take small segment from the audio for further processing.

   Separating the samples into fixed length segments is know as framing or frame blocking. These frames are usually from 5 milliseconds to 100 milliseconds. But in case of speech signal to preserve the phonemes, we often take the length of 20-40 milliseconds, which is usually the length of phonemes with 10-15 milliseconds overlap.

   These segments are later converted to frequency domain with an FFT.

   ![](/images/framing.jpg)

   _Why do we use overlapping of the frames?_

   We can imagine non-overlapping rectangular frame. Each sample is, somehow, treated with the same weight. However, when we process the features extracted from two consecutive frames, the change of property between the frames may induce a discontinuity, or a jump ("the difference of parameter values of neighboring frames can be higher"). This blocking effect can create disturbance in the signal.

   In Python, the array of amplitude is framed as:

   ```python
      frame_size = 0.02
      frame_stride = 0.01 # frame overlap = 0.02 - 0.01 = 0.01 (10ms)
   
      frame_length, frame_step = int(round(frame_size * sr)), int(round(frame_stride * sr))  # Convert from seconds to samples
      signal_length = len(yemp)
   
      num_frames = int(np.ceil(float(signal_length - frame_length) / frame_step))  # Make sure that we have at least 1 frame
   
      pad_signal_length = num_frames * frame_step + frame_length
      z = np.zeros((pad_signal_length - signal_length))
      pad_signal = np.append(yemp, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
   
      indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
      frames = pad_signal[indices.astype(np.int32, copy=False)]
   ```
3. **Windowing**

   Windowing multiplies the samples by a scaling function. This is done to eliminate discontinuities at the edges of the frames. If the function of windows is defined as $w(n)$, 0 < n < N-1 where N is the number of samples per frame, the resulting signal will be;

   $$

   s(n) = x (n) w (n)

   $$  
   Generally hamming windows are used where

   $$  
   w(n) = 0.54 - 0.46 \\cos\\bigg(\\frac{2\\pi n}{N-1}\\bigg)
   $$

   ```python
   	frames = frames * np.hamming(frame_length)
   ```

   _Why do we use windowing?_

   When we are changing signals from time domain using FFT, we cannot perform computations on infinite data points with computers so all signals are cut off at either end. For example, lets say we want to do FFT on pure sine wave. In the frequency domain, we expect a sharp spike on the respective frequency of the sine wave. But when we visualize it, we see ripple-like graph on many frequencies which is not even the frequency of the wave.

   ![FFT Graph](/images/photo_2020-09-24_10-12-12.jpg "FFT Graph")

   When we cut off signal at either end, we are indirectly multiplying our signal by a square window. So there are variety of window functions people have come up with. Hamming window, analytically, is know to optimize the characteristics needed for speech processing.
4. **Fourier Transform**
   Fourier transform (FT) is a mathematical transform that decomposes a function (often a function of time, or a signal) into its constituent frequencies. This is used to analyze frequencies contained in the speech signal. And it also gives the magnitude of each frequency.

   $$
   X(k) = \\sum_{n=0}^{N-1}x(n).e^{-\\frac{2\\pi i kn}{N}}
   $$
   where $ k = 0, 1, 2 ... N-1 $

   Short-time Fourier transform (STFT) converts the 1-dimensional signal from the time domain into the frequency domain by using the frames and applying a discrete Fourier transform to each frames.
   We can now do an N-point FFT on each frame to calculate the frequency spectrum, which is also called Short-Time Fourier-Transform (STFT). STFT provides the time-localized frequency information because in speech signals, frequency components vary over time. Usually we take N=256.

   *Spectrogram*
   A spectrogram is a visual representation of the spectrum of frequencies of a signal as it varies with time. For some end-to-end systems, spectrograms are taken as input. This helps in 3D visualization of the FFT.
   Magnitude of Spectrogram
   $ S_m = \\lvert FFT(x_i) \\rvert^2$
   Power Spectrogram
   $ S_p = \\frac{S_m}{N} $
   Where N is the number of points considered for FFT computation. (typically 256 or 512)

   In Python

   ```python
     NFFT = 512
     magnitude = np.absolute(np.fft.rfft(frames, NFFT))
     pow_frames = magnitude**2/NFFT
   ```
   ![Spectrogram](/images/spec.png "Spectrogram")
   
5. **Mel-Filter Bank**
	The magnitude spectrum is warped according to the Mel scale in order to adapt the frequency resolution to the non-linear properties of the human ear by being more discriminative at lower frequencies and less discriminative at higher frequencies. We can convert between Hz scale and Mel scale using the following equations:
    