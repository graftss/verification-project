import os
import scipy.stats

# combines the elements in a list of lists, into a single list
def combine_lists(list_of_lists):
  return [entry for list in list_of_lists for entry in list]

# list paths of all CSV files in the `datasets` directory
def all_dataset_paths():
  return ['datasets/' + path for path in os.listdir('datasets') if path.endswith('.csv')]

csv_header = 'name,version,name,wmc,dit,noc,cbo,rfc,lcom,ca,ce,npm,lcom3,loc,dam,moa,mfa,cam,ic,cbm,amc,max_cc,avg_cc,bug'

csv_key_list = csv_header.split(',')

bug_index = csv_key_list.index('bug')

def get_metric_indices():
  return range(3, 23)

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

def binaryify_bug_counts(dataset):
  for entry in dataset:
    entry[bug_index] = '0' if entry[bug_index] == '0' else '1'

# finds metrics that are uncorrelated with bug count, and removes them
# from the dataset.
def remove_uncorrelated_metrics(dataset):
  global csv_header, csv_key_list
  uncorrelated_indices = []
  bug_counts = [int(entry[csv_key_list.index('bug')]) for entry in dataset]

  for index in get_metric_indices():
    metric_values = [float(entry[index]) for entry in dataset]
    (coeff, p_value) = scipy.stats.pearsonr(metric_values, bug_counts)

    if p_value > 0.0001:
      uncorrelated_indices.append(index)

  # make sure the indices are in descending order, so they can be deleted
  uncorrelated_indices.sort()
  uncorrelated_indices.reverse()

  for entry in dataset:
    for index in uncorrelated_indices:
      del entry[index]

  # update csv key list and header
  for index in uncorrelated_indices:
    del csv_key_list[index]
  csv_header = ','.join(csv_key_list)

# writes `dataset` as a CSV into `path`
def write_dataset(path, dataset):
  with open(path, 'w') as out:
    out.write(csv_header + '\n')
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

  # recombine the groups, which are now duplicate-free, into
  # a single combined and validated dataset
  combined = combine_lists(grouped.values())

  # transform bug counts into a 0/1 value
  binaryify_bug_counts(combined)

  # remove metrics that don't correlate with bug count from the dataset
  remove_uncorrelated_metrics(combined)

  write_dataset('combined.csv', combined)

if __name__ == '__main__':
  main()
