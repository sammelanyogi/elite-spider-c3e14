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

I will use a sound wave an process it in python to make it more clear.

```python
  import librosa 
  import numpy as np 
  import librosa.display
  import matplotlib.pyplot as plt
  import sounddevice as sd
  y, sr = librosa.load("00a80b4c8a.flac", sr=16000)
```

1. **Pre-emphasis**

   This is the first step in feature generation. In speech production, high frequencies usually have smaller magnitudes compared to lower frequencies. So in order to counter the effect we apply pre-emphasis signal to amply the amplitude of high frequencies and lower the amplitude of lower frequencies.

   If $ x(t) $ is the signal,

   $$
   y(t) = x(t) - \\alpha x(t-1)
   $$

   Where, $\\alpha  $ is generally 0.95 or 0.97.

   ```python
       alpha = 0.97
       y_emp = np.append(y[0], y[1:] - alpha * y[:-1])
   ```

   We can visualize the initial signal and amplified signal.

   ![](/images/waves.png)
2. **Framing**

   Acoustic signal is perpetually changing in speech. 