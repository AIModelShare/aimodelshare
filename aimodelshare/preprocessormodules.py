import os
import zipfile
import sys
import pickle
import tempfile
import dill
import importlib
import inspect
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
    temp_dir=tempfile.gettempdir()

    # Create a ZipFile Object and load sample.zip in it
    with ZipFile(filepath, 'r') as zipObj:
      # Extract all the contents of zip file in current directory
      zipObj.extractall(temp_dir)

    folderpath=os.path.dirname(os.path.abspath(filepath))
    file_name=os.path.basename(filepath)
    import os
    pickle_file_list=[]
    for file in os.listdir(temp_dir):
      if file.endswith(".pkl"):
          pickle_file_list.append(os.path.join(temp_dir, file))
    for i in pickle_file_list: 
      objectname=str(os.path.basename(i)).replace(".pkl","")
      objects={objectname:""}
      globals()[objectname]=pickle.load(open(str(i), "rb" ) )
    # First import preprocessor function to session from preprocessor.py
    exec(open(os.path.join(temp_dir,'preprocessor.py')).read(),globals())
    try:
      # clean up temp directory files for future runs
      os.remove(os.path.join(temp_dir,"preprocessor.py"))
    except:
      pass
    try:
      for i in pickle_file_list: 
          objectname=str(i)+".pkl"
          os.remove(os.path.join(temp_dir,objectname))
    except:
      pass
    return preprocessor

import os

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
      temp_dir=tempfile.gettempdir()
      try:
          os.remove(os.path.join(folderpath,"preprocessor.zip"))
      except:
          pass
      #save function code within temporary folder
      source = inspect.getsource(preprocessor_fxn)
      with open(os.path.join(temp_dir,"preprocessor.py"), "w") as f:
          f.write(source)

      # create a ZipFile object
      zipObj = ZipFile(os.path.join(folderpath,"preprocessor.zip"), 'w')
      # Add preprocessor function to the zipfile
      zipObj.write(os.path.join(temp_dir,"preprocessor.py"),"preprocessor.py")

      #getting list of global variables used in function

      import inspect
      function_objects=list(inspect.getclosurevars(preprocessor_fxn).globals.keys())

      import sys
      modulenames = ["sklearn","keras","tensorflow","cv2","resize","pytorch", "librosa"]
      function_objects_nomodules = [i for i in function_objects if i not in list(modulenames)]

      def savetopickle(function_objects_listelement):
        import pickle
        pickle.dump(globals()[function_objects_listelement], open( os.path.join(temp_dir,function_objects_listelement+".pkl"), "wb" ) )
        return function_objects_listelement

      savedpreprocessorobjectslist = list(map(savetopickle, function_objects_nomodules))

      # take savedpreprocessorobjectslist pkl files saved to tempdir to zipfile
      import pickle
      import string

      
      for i in savedpreprocessorobjectslist: 
          objectname=str(i)+".pkl"
          zipObj.write(os.path.join(temp_dir,objectname),objectname)

      # close the Zip File
      zipObj.close()

      try:
          # clean up temp directory files for future runs
          os.remove(os.path.join(temp_dir,"preprocessor.py"))

          for i in savedpreprocessorobjectslist: 
              objectname=str(i)+".pkl"
              os.remove(os.path.join(temp_dir,objectname))
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

