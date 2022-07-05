import os
import zipfile
import sys
import pickle
import tempfile
import importlib
import inspect
import shutil
from pathlib import Path
#from aimodelshare.python.my_preprocessor import *

# how to import a preprocessor from a zipfile into a tempfile then into the current session
def import_preprocessor(filepath):
    """
    Import preprocessor function to session from zip file 
    Inputs: 1 
    Output: preprocessor function
    
    Parameters:
    -----------
    `filepath`: ``string``
        value - absolute path to preprocessor file 
        [REQUIRED] to be set by the user
        "./preprocessor.zip" 
        file is generated using export_preprocessor function from the AI Modelshare library 
        preprocessor function should always be named 'preprocessor' to work properly in aimodelshare process
    
    Returns:
    --------
    imports preprocessor function to session
    """

    #preprocessor fxn should always be named "preprocessor" to work properly in aimodelshare process.
    import tempfile
    from zipfile import ZipFile
    import inspect
    import os
    import pickle
    import string
    
    #create temporary folder
    temp_dir = tempfile.mkdtemp()

    # Create a ZipFile Object and load sample.zip in it
    with ZipFile(filepath, 'r') as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall(temp_dir)

    folderpath = os.path.dirname(os.path.abspath(filepath))
    file_name = os.path.basename(filepath)

    pickle_file_list = []
    zip_file_list = []
    for file in os.listdir(temp_dir):
        if file.endswith(".pkl"):
            pickle_file_list.append(os.path.join(temp_dir, file))
        if file.endswith(".zip"):
            zip_file_list.append(os.path.join(temp_dir, file))
    
    for i in pickle_file_list: 
        objectname=str(os.path.basename(i)).replace(".pkl", "")
        objects={objectname:""}
        globals()[objectname]=pickle.load(open(str(i), "rb" ) )
    
    # Need spark session and context to instantiate model object
    # zip_file_list is only used by pyspark
    if len(zip_file_list):
        try:
            from pyspark.sql import SparkSession
        except:
            raise("Error: Please install pyspark to enable pyspark features")
            
        spark = SparkSession \
            .builder \
            .appName('Pyspark Model') \
            .getOrCreate()
    
    for i in zip_file_list:
        objectnames = str(os.path.basename(i)).replace(".zip", "").split("__")
        dir_path = i.replace(".zip", "")
        Path(dir_path).mkdir(parents=True, exist_ok=True)
          
        # Create a ZipFile Object and load module.zip in it
        with ZipFile(i, 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(dir_path)

        preprocessor_type = objectnames[0].split("_")[0]
        objectname = objectnames[1]
        from aimodelshare.aimsonnx import pyspark_model_from_string
        preprocessor_class = pyspark_model_from_string(preprocessor_type)
        if preprocessor_type == "PipelineModel":
            preprocessor_model = preprocessor_class(stages=None)
        else:
            preprocessor_model = preprocessor_class()

        preprocessor_model = preprocessor_model.load(dir_path)
        globals()[objectname] = preprocessor_model

    # First import preprocessor function to session from preprocessor.py
    exec(open(os.path.join(temp_dir, 'preprocessor.py')).read(),globals())
    try:
        # clean up temp directory files for future runs
        os.remove(os.path.join(temp_dir, "preprocessor.py"))
    except:
        pass
   
    try:
        for file in pickle_file_list: 
            os.remove(file)
        
        for file in zip_file_list:
            os.remove(file)
    except:
        pass

    return preprocessor

def export_preprocessor(preprocessor_fxn,directory, globs=globals()):
    """
    Exports preprocessor and related objects into zip file for model deployment
    Inputs: 2 
    Output: zipfile named 'preprocessor.zip'
    
    Parameters:
    -----------
    `preprocessor_fxn`: name of preprocessor function
        Preprocessor function should always be named "preprocessor" to work properly in aimodelshare process.
    `directory`: ``string`` folderpath to preprocessor function
        use "" to reference current working directory
    
    Returns:
    --------
    file named 'preprocessor.zip' in the correct format for model deployment
    """
    #preprocessor fxn should always be named "preprocessor" to work properly in aimodelshare process.
    try:
        import tempfile
        from zipfile import ZipFile
        import inspect
        import os

        globals().update(preprocessor_fxn.__globals__)

        folderpath=directory

        #create temporary folder
        temp_dir=tempfile.mkdtemp()
        try:
            os.remove(os.path.join(folderpath, "preprocessor.zip"))
        except:
            pass

        #save function code within temporary folder
        source = inspect.getsource(preprocessor_fxn)
        with open(os.path.join(temp_dir, "preprocessor.py"), "w") as f:
            f.write(source)

        # create a ZipFile object
        zipObj = ZipFile(os.path.join(folderpath, "preprocessor.zip"), 'w')
        # Add preprocessor function to the zipfile
        zipObj.write(os.path.join(temp_dir,"preprocessor.py"),"preprocessor.py")

        #getting list of global variables used in function

        import inspect
        function_objects=list(inspect.getclosurevars(preprocessor_fxn).globals.keys())
        
        import sys
        import imp
        modulenames = ["sklearn","keras","tensorflow","cv2","resize","pytorch","librosa","pyspark"]

        # List all standard libraries not covered by sys.builtin_module_names
        paths = (os.path.abspath(p) for p in sys.path)
        stdlib = {
            p for p in paths
            if p.startswith((sys.prefix)) 
                and 'site-packages' not in p
        }

        # Exclude standard libraries
        for module_name in function_objects:
            try:
                if module_name in sys.builtin_module_names:
                    modulenames.append(module_name)
                    continue

                module_path = imp.find_module(module_name)[1]
                if os.path.dirname(module_path) in stdlib:
                    modulenames.append(module_name)
            except Exception as e:
                # print(e)
                continue

        function_objects_nomodules = [i for i in function_objects if i not in list(modulenames)]

        def savetopickle(function_objects_listelement):
            import pickle
            pickle.dump(globals()[function_objects_listelement], open( os.path.join(temp_dir,function_objects_listelement+".pkl"), "wb" ) )
            return function_objects_listelement

        def save_to_zip(function_objects_listelement):
            model_name_path = str(globals()[function_objects_listelement]) + "__" + function_objects_listelement
            temp_path = os.path.join(temp_dir, model_name_path)
            try:
                shutil.rmtree(temp_path)
            except:
                pass

            if not os.path.exists(temp_path):
                os.mkdir(temp_path)

            globals()[function_objects_listelement].write().overwrite().save(temp_path)

            # calling function to get all file paths in the directory
            from aimodelshare.aimsonnx import get_pyspark_model_files_paths
            file_paths = get_pyspark_model_files_paths(temp_path)

            temp_zip_path = os.path.join(temp_dir, model_name_path + ".zip")
            with ZipFile(temp_zip_path,'w') as zip:
                # writing each file one by one
                for file in file_paths:
                    zip.write(os.path.join(temp_path, file), file)

            # cleanup
            try:
                shutil.rmtree(temp_path)
            except:
                pass

            return model_name_path

        export_methods = []
        savedpreprocessorobjectslist = []
        for function_objects_nomodule in function_objects_nomodules:
            try:
                savedpreprocessorobjectslist.append(savetopickle(function_objects_nomodule))
                export_methods.append("pickle")
            except Exception as e:
                # print(e)
                try:
                    os.remove(os.path.join(temp_dir, function_objects_nomodule+".pkl"))
                except:
                    pass
                # print("Try .zip export approach")
                try:
                    savedpreprocessorobjectslist.append(save_to_zip(function_objects_nomodule))
                    export_methods.append("zip")
                except Exception as e:
                    # print(e)
                    pass
        
        # take savedpreprocessorobjectslist pkl & zip files saved to tempdir to zipfile
        import pickle
        import string


        for i, value in enumerate(savedpreprocessorobjectslist):
            if export_methods[i] == "pickle":
                objectname = str(value) + ".pkl"
            elif export_methods[i] == "zip":
                objectname = str(value) + ".zip"
            zipObj.write(os.path.join(temp_dir, objectname), objectname)

        # close the Zip File
        zipObj.close()

        try:
            # clean up temp directory files for future runs
            os.remove(os.path.join(temp_dir,"preprocessor.py"))

            for i, value in enumerate(savedpreprocessorobjectslist):
                if export_methods[i] == "pickle":
                    objectname = str(value) + ".pkl"
                elif export_methods[i] == "zip":
                    objectname = str(value) + ".zip"
                os.remove(os.path.join(temp_dir, objectname))
        except:
            pass

    except Exception as e:
        print(e)

    return print("Your preprocessor is now saved to 'preprocessor.zip'")

def upload_preprocessor(preprocessor_path, client, bucket, model_id, model_version):

  try:

    
    # Check the preprocessor {{{
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(
            f"The preprocessor file at {preprocessor_path} does not exist"
        )

    
    file_name = os.path.basename(preprocessor_path)
    file_name, file_ext = os.path.splitext(file_name)
    
    from zipfile import ZipFile
    dir_zip = preprocessor_path

    #zipObj = ZipFile(os.path.join("./preprocessor.zip"), 'a')
    #/Users/aishwarya/Downloads/aimodelshare-master
    client["client"].upload_file(dir_zip, bucket, model_id + "/runtime_preprocessor" + ".zip")
  except Exception as e:
    print(e)


     

__all__ = [
    import_preprocessor,
    export_preprocessor,
    upload_preprocessor,
    
]

