% Actual code in practice

pc1_path = 'dataset/csi0526Atake2.dat';
pc2_path = 'dataset/csi0526Btake2.dat';

pc1 = read_bf_file(pc1_path);
pc2 = read_bf_file(pc2_path);

csi_trace = pc1;

csis = zeros(3,30,length(csi_trace));

for i=1:length(csi_trace)
    csis(:,:,i) = get_scaled_csi(csi_trace{i});
end

%plot(db(abs(squeeze(r).')))

%csi_entry = csi_trace_pc1{2}.csi;
%csi_entry = csi_trace_pc2{1}

antenna1 = csis(1,:,:);
antenna2 = csis(2,:,:);
antenna3 = csis(3,:,:);

figure(1);
plot(db(abs(squeeze(antenna1).')));
xlabel('#Package');
ylabel('SNR [dB]');
title('PC2 Antenna1');

figure(2);
plot(db(abs(squeeze(antenna2).')));
xlabel('#Package');
ylabel('SNR [dB]');
title('PC2 Antenna2');

figure(3);
plot(db(abs(squeeze(antenna3).')));
xlabel('#Package');
ylabel('SNR [dB]');
title('PC2 Antenna3');