import os
import json 
import pandas as pd
import csv
from pathlib import Path
import ast
import random


# Used to identify answer text in original dataset. 
answer_index = {
    "A" : 0,
    "B" : 1,
    "C" : 2,
    "D" : 3,
}


def cmu_conversion_format(parent_dir_path:str) -> None:
    """
Function that takes as input the path to either the Train, Dev or Test Folder of the RACE dataset. It then
performs a search down its subdirectories and finds all text files of the RACE dataset. 

Referring to the format used in Yifan Gao's work, we transform the dataset into a large JSON file containing a list 
of dictionaries. Each of the dictionaries has the following keys: article; question; answer_text; distractor. And each key 
holds one element, i.e. for each dictionary you'll have the true answer and a corresponding distractor. 

For the purpose of structure:
    # INPUT (CMU Dataset) keys: answers,options,questions,article,id
    # OUTPUT (Our New Dataset) keys: article, question, distractor, answer_text
    """
    def sample_random_file(parent_dir_path:str) -> dict:
        """
    Helper function that picks a file at random from the RACE dataset and returns its contents.    
        """
        random_file = random.sample([path for path in Path(parent_dir_path).rglob('*.txt')], 1)
        with open(random_file[0], 'r') as f:
            ctnt = f.read()
            content_dictionary = ast.literal_eval(ctnt)
        return content_dictionary

    def remove_elem_from_list_by_id(input_list:list, idx:int)-> list:
        """
    Helper function that removes an element from a list and returns that updated list. 
    eg. [A, B, C, D] -> Remove element at index 1 -> [A, C, D]
        """
        return input_list[:idx] + input_list[idx+1:]

    f_name = "RACE_"+os.path.basename(parent_dir_path) + "_new.json"
    with open(f_name, 'w') as f_out:
        f_out.write('[')
        for sample in Path(parent_dir_path).rglob("*.txt"):
                with open(sample, 'r') as f:
                    sample_content = f.read()
                    sample_dict = ast.literal_eval(sample_content)
                    
                    for i in range(len(sample_dict["questions"])):
                        answer_idx = answer_index[sample_dict["answers"][i]]
                        distractors = remove_elem_from_list_by_id(sample_dict["options"][i], answer_idx)
                        for j in range(len(distractors)):
                            one_distractor_dict = {}
                            one_distractor_dict["article"] = sample_dict["article"].replace(" ", ",")
                            one_distractor_dict["question"] = sample_dict["questions"][i].replace(" ", ",")
                            one_distractor_dict["answer_text"] = sample_dict["options"][i][answer_idx].replace(" ", ",")
                            one_distractor_dict["distractor"] = distractors[j].replace(" ", ",")
                            
                            f_out.write(json.dumps(one_distractor_dict))
                            f_out.write(',\n')
        f_out.write(']')
        f_out.close()



def main():
    cmu_conversion_format("RACE/dev")
    cmu_conversion_format("RACE/test")
    cmu_conversion_format("RACE/train")

if __name__ == "__main__":
    main()