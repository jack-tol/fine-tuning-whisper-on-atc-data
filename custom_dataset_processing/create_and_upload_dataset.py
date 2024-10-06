from datasets import load_dataset, concatenate_datasets, DatasetDict

dataset1 = load_dataset('Jzuluaga/atco2_corpus_1h', split='test')
dataset2_train = load_dataset('Jzuluaga/uwb_atcc', split='train')
dataset2_test = load_dataset('Jzuluaga/uwb_atcc', split='test')

combined_dataset = concatenate_datasets([dataset1, dataset2_train, dataset2_test])

shuffled_dataset = combined_dataset.shuffle(seed=42)

train_test_split = shuffled_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

columns_to_remove = ['id', 'segment_start_time', 'segment_end_time', 'duration']
train_dataset = train_dataset.remove_columns(columns_to_remove)
test_dataset = test_dataset.remove_columns(columns_to_remove)

texts_to_remove = [
    "ing echo then direct direct", "one", "standby", "is now ready for depa", "roger", "praha mike",
    "calibra", "romeo", "three nine zer", "praha", "two zero push approved", "merci au plaisir aurevoir",
    "boeing", "praha luftha", "four five", "decimal three", "ils one three", "sky travel", "cali",
    "course omelo", "maintain present heading one descend five thousand", "fedex five six", "tower five one",
    "one five zero two", "follow airbus", "black sea", "ils", "thomson one zero alfa", "when ready",
    "calibra ground", "our seven five", "to the left and the ground", "potvrzuju", "expedite", "irm",
    "lufthansa four tango papa", "push app", "seven nine o", "india november", "three eight whiskey ground",
    "medium", "oscar india november", "flight level", "tower oscar kilo", "with high speed", "oscar yankee romeo november",
    "and hold two two", "vfr traffic", "cleared to land", "two is holding short", "runway one", "sky"
]

def filter_samples(sample):
    return sample['text'] not in texts_to_remove

train_dataset = train_dataset.filter(filter_samples)
test_dataset = test_dataset.filter(filter_samples)

final_dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

final_dataset.push_to_hub("atc_dataset")