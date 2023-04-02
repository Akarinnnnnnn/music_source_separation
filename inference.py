import os
import time

import torch
import soundfile as sf

from models.resunet_subbandtime import ResUNet143_Subbandtime
from separator import Separator


def user_defined_build_separator() -> Separator:
    r"""Users could modify this file to load different models.

    Returns:
        separator: Separator
    """

    input_channels = 2
    output_channels = 2
    target_sources_num = 1
    segment_samples = int(44100 * 30.)
    batch_size = 1
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model_type = "Vocal"

    if model_type == "Inst":
        checkpoint_path = os.path.join("pths", 
            "resunet143_subbtandtime_accompaniment_16.4dB_500k_steps_v2.pth")
    elif model_type == "Vocal":
        checkpoint_path = os.path.join("pths", 
            "resunet143_subbtandtime_vocals_8.7dB_500k_steps_v2.pth")
    

    # Create model.
    model = ResUNet143_Subbandtime(
        input_channels=input_channels,
        output_channels=output_channels,
        target_sources_num=target_sources_num,
    )

    # Load checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint["model"])

    # Move model to device.
    model.to(device)

    # Create separator.
    separator = Separator(
        model=model,
        segment_samples=segment_samples,
        batch_size=batch_size,
        device=device,
    )

    return separator


def main():
    r"""An example of using bytesep in your programme. After installing bytesep, 
    users could copy and execute this file in any directory.
    """

    # Build separator.
    separator = user_defined_build_separator()

    audio, sr = sf.read("todo.wav")

    # dummy audio
    input_dict = {'waveform': audio.T}

    # Separate.
    separate_time = time.time()
    sep_audio = separator.separate(input_dict)

    # print(sep_audio.shape)

    sf.write("done.wav", sep_audio.T, samplerate=sr)

    print("Done! {:.3f} s".format(time.time() - separate_time))


if __name__ == "__main__":
    main()