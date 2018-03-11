    words={k:word.strip() for k,word in enumerate(islice(open(config.words_path),config.max_vocab_size))}
    tags={k:tag.strip() for k,tag in enumerate(open(config.tags_path))}
    target_processing = sequence_line_processing(tags, max_len=config.max_tags, return_length=False)
    tags_size=len(target_processing.vocab)
    weight_processing = data_line_processing(max_len=config.max_tags)
    text_processing = sequence_line_processing(words, max_len=config.sen_len, return_length=True)
    vocab_size=len(text_processing.vocab)
    # datas
    line_processing = json_line_processing(
            OrderedDict((("tags",target_processing),("weights",weight_processing),("text",text_processing))))
    train_data = LineBasedDataset(config.train_data_path, line_processing, batch_size= config.batch_size) 
    dev_data = LineBasedDataset(config.dev_data_path, line_processing, batch_size = config.batch_size)
    test_datas = [LineBasedDataset(path, line_processing, batch_size = config.batch_size)
        for path in config.test_data_paths]
