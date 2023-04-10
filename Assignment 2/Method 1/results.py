
class TextData:
  def __init__(self, text):
    lines = text.split("\n")
    model_name_line = lines[0].strip().split(': ')[1]
    if "(" in model_name_line:
        description_start = model_name_line.index("(")
        self.model = model_name_line[:description_start-1]
        self.has_description = "(description)" in model_name_line
    else:
        self.model = model_name_line

    for line in lines[1:]:
      key, value, label, *_ = line.split()
      setattr(self, key, label)

def read_results(filename):
  # Open the file and read its contents
  with open(filename,'r', encoding='utf-16') as f:
      contents = f.read()
  # Split the contents into sections for each model
  sections = contents.split("---")[1:-1]

  # Create a TextData object for each section and store it in a list
  models = []
  for section in sections:
      model_data = TextData(section.strip())
      models.append(model_data)
  return models

all = read_results("transformers.txt")
all_description = read_results("transformers-descriptions.txt")
sgpt = read_results("sgpt-test.txt")
sgpt_description = read_results("sgpt-descriptions.txt")

# combine the lists
all.extend(sgpt)
all_description.extend(sgpt_description)

all_map = {x.model: x for x in all}
all_description_map = {x.model: x for x in all_description}

n = 10

print(f'---\nTop {n} models by MAP:')
# compare the results
for x, model in enumerate(sorted(all, key=lambda x: x.map, reverse=True)[0:n]):
  print(f"{x+1}. {model.model}: {model.map} (w/description: {all_description_map[model.model].map})")

print(f'---\nTop {n} models by P@10:')
for x, model in enumerate(sorted(all, key=lambda x: x.P_10, reverse=True)[0:n]):
  print(f"{x+1}. {model.model}: {model.P_10} (w/description: {all_description_map[model.model].P_10})")

# print the results in a markdown table, including the rank, and show the percentage difference between without and with description
print(f'---\nTop {n} models by MAP:')
print('| Rank | Model | MAP | MAP (w/description) | Difference |')
print('| --- | --- | --- | --- | --- |')
for x, model in enumerate(sorted(all, key=lambda x: x.map, reverse=True)[0:n]):
  print(f"| {x+1} | {model.model} | {model.map} | {all_description_map[model.model].map} | {round((float(model.map) - float(all_description_map[model.model].map)) / float(model.map) * 100, 2) * -1}% |")

# print the results in a markdown table, including the rank, and show the percentage difference between with and without description
print(f'---\nTop {n} models by P@10:')
print('| Rank | Model | P@10 | P@10 (w/description) | Difference |')
print('| --- | --- | --- | --- | --- |')
for x, model in enumerate(sorted(all, key=lambda x: x.P_10, reverse=True)[0:n]):
  print(f"| {x+1} | {model.model} | {model.P_10} | {all_description_map[model.model].P_10} | {round((float(model.P_10) - float(all_description_map[model.model].P_10)) / float(model.P_10) * 100, 2) * -1}% |")

# print the rest of the results in a markdown table, including the rank
print('---\nRest of the models by MAP and P@10:')
print('| Rank | Model | MAP | MAP (w/description) | P@10 | P@10 (w/description) |')
print('| --- | --- | --- | --- | --- | --- |')
for x, model in enumerate(sorted(all, key=lambda x: x.map, reverse=True)[n:]):
  print(f"| {x+n+1} | {model.model} | {model.map} | {all_description_map[model.model].map} | {model.P_10} | {all_description_map[model.model].P_10} |")

# calcuate the average difference between with and without description in the top 10 models in percentage
print('---\nAverage difference between with and without description in the top 10 models:')
print(f"MAP: {round(sum([round((float(model.map) - float(all_description_map[model.model].map)) / float(model.map) * 100, 2) * -1 for model in sorted(all, key=lambda x: x.map, reverse=True)[0:n]]) / n, 2)}%")
print(f"P@10: {round(sum([round((float(model.P_10) - float(all_description_map[model.model].P_10)) / float(model.P_10) * 100, 2) * -1 for model in sorted(all, key=lambda x: x.P_10, reverse=True)[0:n]]) / n, 2)}%")