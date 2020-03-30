import os

# combines the elements in a list of lists, into a single list
def combine_lists(list_of_lists):
  return [entry for list in list_of_lists for entry in list]

# list paths of all CSV files in the `datasets` directory
def all_dataset_paths():
  return ['datasets/' + path for path in os.listdir('datasets') if path.endswith('.csv')]

# read and parse a CSV file in the `data` directory
def parse_dataset(path):
  with open(path) as f:
    # ignore the list of fields from first line
    fields = f.readline().strip().split(',')
    lines = f.read().splitlines()

  return [line.split(',') for line in lines]

# groups `dataset` entries by the WMC metric value
# e.g. `result['0'] will be the list of entries where WMC = 0
def group_dataset(dataset):
  result = {}

  for entry in dataset:
    key = entry[3]
    if key not in result:
      result[key] = []
    result[key].append(entry)

  return result

# checks equality for the values of the 20 metrics
# between the two dataset entries `a` and `b`
def equal_entries(a, b):
  for idx in range(3, 23):
    if a[idx] != b[idx]:
      return False
  return True

def remove_duplicates(dataset):
  n = len(dataset)
  dupe_indices = []

  for i in range(n):
    # if the entry has already been found to be a duplicate,
    # we don't need to check if there are duplicates of it
    if i in dupe_indices:
      continue

    # check following elements for duplicates
    for j in range(i + 1, n):
      if equal_entries(dataset[i], dataset[j]):
        dupe_indices.append(j)

  # find unique dupe indices and sort them into descending order
  dupe_indices = list(set(dupe_indices))
  dupe_indices.sort(reverse=True)

  for idx in dupe_indices:
    del dataset[idx]

# writes `dataset` as a CSV into `path`
def write_dataset(path, dataset):
  with open(path, 'w') as out:
    # write the field header line manually
    out.write('name,version,name,wmc,dit,noc,cbo,rfc,lcom,ca,ce,npm,lcom3,loc,dam,moa,mfa,cam,ic,cbm,amc,max_cc,avg_cc,bug')
    for entry in dataset:
      out.write(','.join(entry) + '\n')

def main():
  datasets = [parse_dataset(path) for path in all_dataset_paths()]
  combined = combine_lists(datasets)

  # group dataset by first metric; we know that duplicate entries
  # have to agree in their first metric, so we only need to check
  # for duplicates within each group
  grouped = group_dataset(combined)
  for k, group in grouped.items():
    remove_duplicates(group)

  validated = combine_lists(grouped.values())
  write_dataset('combined.csv', validated)

if __name__ == '__main__':
  main()
