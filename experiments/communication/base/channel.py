"""Unit-energy QPSK, Sionna AWGN/TDL, and explicit packet erasures."""
from __future__ import annotations

import numpy as np


def qpsk_symbols(bits):
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    pairs = np.pad(bits, (0, len(bits) % 2)).reshape(-1, 2)
    return ((1-2*pairs[:, 0].astype(np.float32)) +
            1j*(1-2*pairs[:, 1].astype(np.float32))) / np.sqrt(2.)


def hard_qpsk(symbols):
    return np.column_stack([np.real(symbols) < 0, np.imag(symbols) < 0]).astype(np.uint8).reshape(-1)


def soft_qpsk(symbols, noise_variance, gain=1.):
    # Sionna decoder input is log P(bit=1)/P(bit=0).
    return (-2*np.sqrt(2.)*gain/max(float(noise_variance), 1e-12) *
            np.column_stack([np.real(symbols), np.imag(symbols)])).reshape(-1)


class Channels:
    def __init__(self, config):
        import torch
        import sionna.phy
        from sionna.phy.channel import AWGN
        from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
        self.torch, self.sionna, self.config = torch, sionna, config
        self.awgn = AWGN(device="cpu")
        self.k, self.n = config["ldpc_k"], config["ldpc_n"]
        self.encoder = LDPC5GEncoder(self.k, self.n, device="cpu")
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True, return_infobits=True,
                                    num_iter=config["ldpc_iterations"], device="cpu")
        self.time_channels = {}

    def physical(self, symbols, channel, snr_db, seed):
        torch = self.torch
        self.sionna.phy.config.seed = int(seed)
        variance = 10.**(-float(snr_db)/10.)
        x = torch.tensor(symbols, dtype=torch.complex64, device="cpu")
        if channel == "awgn":
            y = self.awgn(x, torch.tensor(variance, dtype=torch.float32))
            return y.detach().cpu().numpy(), variance, 1.
        if channel != "fading":
            raise ValueError(channel)
        from sionna.phy.channel import TimeChannel
        from sionna.phy.channel.tr38901 import TDL
        c = self.config
        length = len(symbols)
        if length not in self.time_channels:
            model = TDL(c["tdl_model"], delay_spread=c["delay_spread_seconds"],
                        carrier_frequency=c["carrier_frequency_hz"], min_speed=0., max_speed=0.,
                        num_rx_ant=1, num_tx_ant=1, device="cpu")
            self.time_channels[length] = TimeChannel(model, bandwidth=c["bandwidth_hz"],
                num_time_samples=length, l_min=c["l_min"], l_max=c["l_max"],
                normalize_channel=True, return_channel=True, device="cpu")
            if len(self.time_channels) > 8:
                self.time_channels.pop(next(iter(self.time_channels)))
        y, h = self.time_channels[length](x.reshape(1, 1, 1, -1), torch.tensor(variance))
        received = y.reshape(-1).detach().cpu().numpy()
        taps = h.reshape(-1, c["l_max"]-c["l_min"]+1).detach().cpu().numpy().mean(axis=0)
        n_fft = 1 << (len(received)+len(taps)-2).bit_length()
        response = np.fft.fft(taps, n_fft)
        equalizer = np.conj(response)/(np.abs(response)**2+variance)
        equalized = np.fft.ifft(np.fft.fft(received, n_fft)*equalizer)[:length]
        transfer = equalizer*response
        gain = float(np.real(transfer.mean()))
        residual_isi = max(0., float(np.mean(np.abs(transfer)**2))-gain**2)
        post_noise = variance*float(np.mean(np.abs(equalizer)**2))
        return equalized, post_noise+residual_isi, gain

    def transmit(self, bits, channel, coding, value, seed):
        bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
        if not len(bits):
            raise ValueError("Empty transmission")
        if coding not in {"uncoded", "ldpc"}:
            raise ValueError(coding)
        torch = self.torch
        blocks = int(np.ceil(len(bits)/self.k)) if coding == "ldpc" else 0
        with torch.inference_mode():
            if coding == "ldpc":
                source = np.pad(bits, (0, blocks*self.k-len(bits))).reshape(blocks, self.k)
                encoded = self.encoder(torch.tensor(source, dtype=torch.float32)).cpu().numpy().astype(np.uint8).reshape(-1)
            else:
                encoded = bits.copy()
            symbols = qpsk_symbols(encoded)
            erased = np.zeros(len(encoded), dtype=bool)
            lost_packets = np.zeros(0, dtype=bool)
            if channel == "packet_loss":
                if not 0 <= float(value) <= 1:
                    raise ValueError("Invalid packet loss rate")
                size = self.config["packet_size_bits"]
                lost_packets = np.random.default_rng(seed).random(int(np.ceil(len(encoded)/size))) < value
                erased = np.repeat(lost_packets, size)[:len(encoded)]
                # Ideal surviving QPSK symbols: no additional AWGN in this experiment.
                raw = hard_qpsk(symbols)[:len(encoded)]
                raw[erased] = 0
                llr = np.where(encoded, 20., -20.).astype(np.float32)
                llr[erased] = 0.
            else:
                y, variance, gain = self.physical(symbols, channel, value, seed)
                raw = hard_qpsk(y)[:len(encoded)]
                llr = soft_qpsk(y, variance, gain)[:len(encoded)]
            if coding == "ldpc":
                decoded = self.decoder(torch.tensor(llr.reshape(blocks, self.n), dtype=torch.float32))
                decoded = decoded.cpu().numpy().astype(np.uint8).reshape(-1)[:len(bits)]
            else:
                decoded = raw[:len(bits)]
        stats = {"payload_bits": len(bits), "transmitted_bits": 2*len(symbols),
                 "qpsk_symbols": len(symbols), "ldpc_blocks": blocks,
                 "padding_information_bits": blocks*self.k-len(bits) if blocks else 0,
                 "raw_coded_bit_error_rate": float(np.mean(raw != encoded)),
                 "decoded_information_ber": float(np.mean(decoded != bits)),
                 "erased_bit_fraction": float(erased.mean()),
                 "packet_count": len(lost_packets), "lost_packets": int(lost_packets.sum()),
                 "actual_packet_loss_fraction": float(lost_packets.mean()) if len(lost_packets) else 0.}
        return decoded, stats
