import os
import zipfile
import sys
import pickle
import tempfile
import dill
import importlib
import inspect
#from aimodelshare.python.my_preprocessor import *

def export_eval_metric(eval_metric_fxn, directory, name='eval_metric', globs=globals()):
    """
    Exports evaluation metric and related objects into zip file for model deployment
    Inputs: 2 
    Output: zipfile named 'eval_metric.zip'
    
    Parameters:
    -----------
    `eval_metric_fxn`: name of eval metric function
    `directory`: ``string`` folderpath to eval metric function
        use "" to reference current working directory
    `name`: name of the custom eval metric 
    
    Returns:
    --------
    file named 'name.zip' in the correct format for model deployment
    """
    #evaluation metric fxn should always be named "eval_metric" to work properly in aimodelshare process.

    try:
      import tempfile
      from zipfile import ZipFile
      import inspect
      import os

      globals().update(eval_metric_fxn.__globals__)

      folderpath=directory

      #create temporary folder
      temp_dir=tempfile.gettempdir()
      try:
          os.remove(os.path.join(folderpath, name+".zip"))
      except:
          pass
      #save function code within temporary folder
      source = inspect.getsource(eval_metric_fxn)
      with open(os.path.join(temp_dir,"{}.py".format(name)), "w") as f:
          f.write(source)

      # create a ZipFile object
      zipObj = ZipFile(os.path.join(folderpath,"{}.zip".format(name)), 'w')
      # Add preprocessor function to the zipfile
      zipObj.write(os.path.join(temp_dir,"{}.py".format(name)),"{}.py".format(name))

      #getting list of global variables used in function

      import inspect
      function_objects=list(inspect.getclosurevars(eval_metric_fxn).globals.keys())

      import sys
      modulenames = ["sklearn","keras","tensorflow","cv2","resize","pytorch", "librosa"]
      function_objects_nomodules = [i for i in function_objects if i not in list(modulenames)]

      def savetopickle(function_objects_listelement):
        import pickle
        pickle.dump(globals()[function_objects_listelement], open( os.path.join(temp_dir,function_objects_listelement+".pkl"), "wb" ) )
        return function_objects_listelement

      savedevalmetricobjectslist = list(map(savetopickle, function_objects_nomodules))

      # take savedpreprocessorobjectslist pkl files saved to tempdir to zipfile
      import pickle
      import string

      for i in savedevalmetricobjectslist: 
          objectname=str(i)+".pkl"
          zipObj.write(os.path.join(temp_dir,objectname),objectname)

      # close the Zip File
      zipObj.close()

      try:
          # clean up temp directory files for future runs
          os.remove(os.path.join(temp_dir,"{}.py".format(name)))

          for i in savedevalmetricobjectslist: 
              objectname=str(i)+".pkl"
              os.remove(os.path.join(temp_dir,objectname))
      except:
          pass

    except Exception as e:
        print(e)

    return print("Your eval_metric is now saved to '{}.zip'".format(name))

   

__all__ = [
    export_eval_metric    
]

