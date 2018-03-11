function BuildVocabList()
%BUILDVOCABLIST reads the spam emails and preprocess them into seprate words and
% count their frequencies. If the frequency of a word exceeds a cutoff limit, 
% it will be written in the file.
    dirinfo = dir('spam_2/0*');
    freq = containers.Map;
    for i = 1:length(dirinfo)
        filename = dirinfo(i).name;
        email_contents = readFile(strcat('spam_2/', filename));
        % same code from processEmail, but we will include all words
        % and count their frequencies.
        email_contents = lower(email_contents);
        email_contents = regexprep(email_contents, '<[^<>]+>', ' ');
        email_contents = regexprep(email_contents, '[0-9]+', 'number');
        email_contents = regexprep(email_contents, ...
                                    '(http|https)://[^\s]*', 'httpaddr');
        email_contents = regexprep(email_contents, '[^\s]+@[^\s]+', 'emailaddr');
        email_contents = regexprep(email_contents, '[$]+', 'dollar');
        while ~isempty(email_contents)
            [str, email_contents] = ...
                strtok(email_contents, ...
                    [' @$/#.-:&*+=[]?!(){},''">_<;%' char(10) char(13)]);
            str = regexprep(str, '[^a-zA-Z0-9]', '');
            try str = porterStemmer(strtrim(str));
            catch str = ''; continue;
            end;
            if length(str) < 1
                continue;
            end
            try
                freq(str) = freq(str)+1;
            catch
                freq(str) = 0;
            end
    end
    keys = freq.keys;
    values = freq.values;
    cutoff = 100;
    fid = fopen('myvocab.txt', 'wt');
    count = 0;
    for i = 1:length(keys)
        key = keys{i};
        if freq(key) >= cutoff
            count = count+1;
            fprintf(fid, '%-10d%s\n', count, key);
        end
    end
    fclose(fid);

end
