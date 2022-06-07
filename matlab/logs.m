% Getting useful logs

function logs(data)
    csi_trace = read_bf_file(data);
    len = length(csi_trace);
    per = csi_trace{1}.perm;
    time1 = csi_trace{1}.timestamp_low;
    time2 = csi_trace{len}.timestamp_low;
    du = time2 - time1;
    sprintf('len = %d\n perm = %d%d%d\n start = %d\n end = %d \n duration = %d\n', ...
        len, per, time1, time2, du)

end
