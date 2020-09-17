---
title: All of MFCC
date: 2020-09-07
thumb_image: images/12.jpg
image: images/12.jpg
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
   y(t) = x(t) - \alpha x(t-1)
   $$

   where, $\alpha$ is generally 0.95 or 0.97.

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

   **Why do we use overlapping of the frames?**

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

   Windowing multiplies the samples by a scaling function.