import os


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

all = read_results("sgpt-test.txt")
all_description = read_results("sgpt-descriptions.txt")
all_map = {x.model: x for x in all}
all_description_map = {x.model: x for x in all_description}

print('---\nTop 5 models by MAP:')
# compare the results
for x, model in enumerate(sorted(all, key=lambda x: x.map, reverse=True)[0:5]):
  print(f"{x+1}. {model.model}: {model.map} (description: {all_description_map[model.model].map})")

print('---\nTop 5 models by P@10:')
for x, model in enumerate(sorted(all, key=lambda x: x.P_10, reverse=True)[0:5]):
  print(f"{x+1}. {model.model}: {model.P_10} (description: {all_description_map[model.model].P_10})")