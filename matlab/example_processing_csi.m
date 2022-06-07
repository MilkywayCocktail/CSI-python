% Common operations to CSI

csi_trace = read_bf_file('sample_data/log.all_csi.6.7.6');
csi_entry = csi_trace{1};
csi = get_scaled_csi(csi_entry);

plot(db(abs(squeeze(csi).')))
legend('RX Antenna A', 'RX Antenna B', 'RX Antenna C', 'Location', 'SouthEast' );
xlabel('Subcarrier index');
ylabel('SNR [dB]');

csi_entry = csi_trace{20};
csi = get_scaled_csi(csi_entry);
db(get_eff_SNRs(csi), 'pow')