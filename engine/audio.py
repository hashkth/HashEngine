
from .data import *


class Audio:

    core = None
    device = None
    context = None

    # Containes al.AudioData instances for each loaded audio file
    audio_data = {}

    # Contains al.Buffer container for each loaded audio file
    audio_buffers = {}
    
    # Maintains label --> filepath mapping
    loaded = {}

    # Maintains label --> decoded PCM data mapping
    decoded = {}
    
    free_sources = [] # Sources available for playing audio
    busy_sources = {} # Sources currently playing audio: label --> source
    num_sources = 16  # Number of sources created
    
    streams = {}
    
    frame_size = 4096 * 4
    DB_MIN = -80
    DB_MAX = 0

    # Keeps track of when to update source containers
    _check_counter = 0

    @classmethod
    def init(cls, core):
        """Initialize Audio"""
        cls.core = core
        al.init()
        cls.device = al.Device()
        cls.ctx = al.Context(cls.device)
        cls.free_sources = [al.Source() for i in range(cls.num_sources)]

    @classmethod
    def add(cls, label: str, filepath: str, surround: bool = False):
        """Load a new audio under label"""
        if label not in cls.loaded:
            cls.loaded[label] = filepath
            cls.audio_data[label] = al.AudioData(str(filepath), surround)
            cls.audio_buffers[label] = al.Buffer(cls.audio_data[label])

    @classmethod
    def remove(cls, label: str):
        """Remove the audio referenced by label"""
        if label not in cls.loaded:
            return
        if label in cls.busy_sources:
            cls.busy_sources[label].reset()
            cls.free_sources.append(cls.busy_sources.pop(label))
        if label in cls.decoded:
            del cls.decoded[label]
        del cls.audio_data[label]
        del cls.audio_buffers[label]
        del cls.loaded[label]
    
    @classmethod
    def get_source(cls, label: str):
        if label in cls.busy_sources:
            return cls.busy_sources[label]
        
    @classmethod
    def play(cls, label: str, looping: bool = False):
        """Play the audio referenced by label"""
        if label in cls.busy_sources:
            cls.busy_sources[label].play()
            return
        if cls.free_sources:
            cls.free_sources[0].buffer = cls.audio_buffers[label]
            cls.free_sources[0].looping = looping
            cls.free_sources[0].play()
            cls.busy_sources[label] = cls.free_sources.pop(0)

    @classmethod
    def pause(cls, label: str):
        """Pause the audio referenced by label"""
        if label in cls.busy_sources:
            cls.busy_sources[label].pause()
    
    @classmethod
    def stop(cls, label: str):
        """Stop the audio referenced by label"""
        if label in cls.busy_sources:
            cls.busy_sources[label].reset()
            cls.free_sources.append(cls.busy_sources.pop(label))

    @classmethod
    def set_looping(cls, label: str, looping: bool):
        """Modify looping behavior while audio is playing"""
        if label in cls.busy_sources:
            cls.busy_sources[label].looping = looping
    
    @classmethod
    def get_looping(cls, label: str):
        """Get current looping status while audio is playing"""
        if label in cls.busy_sources:
            return cls.busy_sources[label].looping
        
    @classmethod
    def set_offset(cls, label: str, offset: float):
        """Set offset for the audio referenced by label"""
        if label in cls.busy_sources:
            cls.busy_sources[label].offset = offset
    
    @classmethod
    def get_offset(cls, label: str):
        """Get offset for the audio referenced by label"""
        if label in cls.busy_sources:
            return cls.busy_sources[label].offset
        return 0
    
    @classmethod
    def get_audio_data(cls, label: str):
        """Returns info about the audio referenced by label"""
        if label in cls.audio_data:
            return cls.audio_data[label]
        
    @classmethod
    def get_pcm_bytes(cls, label: str):
        """Returns PCM bytes for the audio referenced by label"""
        if label in cls.audio_data:
            return cls.audio_data[label].decode()
        
    @classmethod
    def add_stream(cls, label: str, filepath: str):
        """Load a streamable audio file referred by label"""
        cls.streams[label] = al.Stream(str(filepath))

    @classmethod
    def remove_stream(cls, label: str):
        """Remove the streamable audio file referred by label"""
        if label in cls.streams:
            del cls.streams[label]

    @classmethod
    def get_stream(cls, label: str):
        if label in cls.streams:
            return cls.streams[label]

    @classmethod
    def play_stream(cls, label: str, looping: bool = False):
        """Play the stream referred by label"""
        if label in cls.streams:
            cls.streams[label].looping = looping
            cls.streams[label].play()

    @classmethod
    def pause_stream(cls, label: str):
        """Pause the stream referred by label"""
        if label in cls.streams:
            cls.streams[label].pause()
    
    @classmethod
    def stop_stream(cls, label: str):
        """Stop the stream referred by label"""
        if label in cls.streams:
            cls.streams[label].stop()

    @classmethod
    def set_stream_looping(cls, label: str, looping: bool):
        """Set looping behavior for the stream referred by label"""
        if label in cls.streams:
            cls.streams[label].looping = looping
    
    @classmethod
    def get_stream_looping(cls, label: str):
        """Get current looping status for the stream referred by label"""
        if label in cls.streams:
            return cls.streams[label].looping

    @classmethod
    def set_stream_offset(cls, label: str, offset: float):
        """Set offset for the stream referred by label"""
        if label in cls.streams:
            cls.streams[label].offset = offset

    @classmethod
    def get_stream_offset(cls, label: str):
        """Get offset for the stream referred by label"""
        if label in cls.streams:
            return cls.streams[label].offset
        return 0

    @classmethod
    def compute_spectrum(cls, chunk, sample_rate):
        # Applies FFT to given chunk and returns the spectrum and freqs
        chunk = chunk.astype(np.float32) / 32768.0
        chunk -= np.mean(chunk)
        window = np.hanning(len(chunk))
        chunk *= window
        mag = np.abs(np.fft.rfft(chunk)) / len(chunk)
        spectrum = 10 * np.log10(mag**2 + 1e-12)
        spectrum[:3] = -120.0
        freqs = np.fft.rfftfreq(len(chunk), 1 / sample_rate)
        return freqs, spectrum

    @classmethod
    def make_bars(cls, spectrum, freqs, num_bars):
        # Converts the spectrum into logarithmically-spaced bars
        num_bars = max(16, min(num_bars, 128))
        bars = []
        min_freq, max_freq = 120, 20000
        log_min, log_max = np.log10(min_freq), np.log10(max_freq)
        log_bins = np.logspace(log_min, log_max, num_bars+1, base=10)
        for i in range(num_bars):
            f1, f2 = log_bins[i], log_bins[i+1]
            mask = (freqs >= f1) & (freqs < f2) & (freqs > 0) 
            if np.any(mask):
                bars.append(np.percentile(spectrum[mask], 85))
            else:
                bars.append(cls.DB_MIN)
        return np.array(bars)

    @classmethod
    def get_spectrum(cls, label: str, offset: float, bars: int):
        data = cls.get_audio_data(label)
        if label not in cls.decoded:
            cls.decoded[label] = data.decode()
        bytes_per_frame = data.bytes_per_sample * data.channels
        sample_index = int(offset * data.sample_rate) % data.total_samples
        start_byte = sample_index * bytes_per_frame
        end_byte = start_byte + cls.frame_size * bytes_per_frame
        total_bytes = len(cls.decoded[label])
        if end_byte <= total_bytes:
            raw_slice = cls.decoded[label][start_byte:end_byte]
        else:
            part1 = cls.decoded[label][start_byte:]
            part2 = cls.decoded[label][:end_byte % total_bytes]
            raw_slice = part1 + part2
        final_data = np.frombuffer(raw_slice, dtype=np.int16)
        if data.channels == 2:
            final_data = final_data.reshape(-1, 2)
            final_data = final_data.mean(axis=1).astype(np.int16)
        chunk = final_data
        freqs, spectrum = cls.compute_spectrum(chunk, data.sample_rate)
        out_spectrum = cls.make_bars(spectrum, freqs, bars)
        out_spectrum = np.clip((out_spectrum - cls.DB_MIN) / (cls.DB_MAX - cls.DB_MIN), 0.0, 1.0)
        return out_spectrum

    @classmethod
    def process(cls):
        # Update all streams every frame
        for label in cls.streams:
            cls.streams[label].update()
        
        # Check for freed sources every 60 frames
        if cls._check_counter >= 60:
            remove = []
            for label in cls.busy_sources:
                if not cls.busy_sources[label].playing and not cls.busy_sources[label].paused:
                    cls.busy_sources[label].reset()
                    cls.free_sources.append(cls.busy_sources[label])
                    remove.append(label)
            for label in remove:
                del cls.busy_sources[label]
            cls._check_counter = 0
        cls._check_counter += 1