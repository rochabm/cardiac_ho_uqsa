function write_data_to_txt(filename, header, data)
% Write a cell array of data to a txt file with the given filename and header.
% The header should be a cell array of strings, and the data should be a cell
% array of numeric or string values.

fid = fopen(filename, 'w');
fprintf(fid, '%s,', header{1:end-1});
fprintf(fid, '%s\n', header{end});
for i = 1:size(data,1)
    t=data(i,:);

    fprintf(fid, '%s\n', num2str(data(i,:)));
end
fclose(fid);