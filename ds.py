from glob import glob
from torch.utils.data import Dataset
from pathlib import Path
from torchaudio import load


class WavGlobDataset(Dataset):
    """
    Dataset that loads all WAV files from a directory. It will partition the audio file
    into a specified number of chunks so that it does not overload the memory.
    """
    def __init__(self, d: Path, num_chunks: int = 1):
        if isinstance(d, str):
            d = Path(d)
        wavs = glob(str(d / '*.wav'))
        self.wavs = wavs
        self.num_chunks = num_chunks

    def __len__(self):
        return len(self.wavs)*self.num_chunks

    def __getitem__(self, idx):
        file_idx = idx // self.num_chunks
        chunk_idx = idx % self.num_chunks
        wav = self.wavs[file_idx]
        waveform, sample_rate = load(wav)
        chunks = self.chunk_waveform(waveform)
        return chunks[chunk_idx]

    def chunk_waveform(self, waveform):
        """
        Chunk a waveform into self.num_chunks chunks.
        :param waveform: torch.Tensor (channels, samples)
        """
        return waveform.chunk(self.num_chunks, dim=1)
