# Instructions 

## Path vizualisation

Firt you have to transform the csv file to a folder of plt files for each taxi. You do that by running this command from the root
```
uv run part1/eda/path_vizualization/csv_to_plt_format.py 
```

Then you can vizualise it by running this command from the root
```
uv run part1/eda/path_vizualization/visualize_paths.py
```

Both the csv to plt transformationa and the vizualisation takes quite a bit fo time, so it can be nice to save the path visualization as a pickle file. One does that by choosing the ***Save to pickle file*** in the path vizualisation menu. After you have saved the pickle file, you can quick start the path vizualisation by running this command from root
```
uv run part1/eda/path_vizualization/quick_start.py  
```