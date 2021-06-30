import os
import zipfile
import sys
import pickle
import tempfile
import dill
import importlib
import inspect

# how to import a postprocessor from a zipfile into a tempfile then into the current session
def import_postprocessor(filepath):
      #postprocessor fxn should always be named "postprocessor" to work properly in aimodelshare process.
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
      # First import postprocessor function to session from postprocessor.py
      exec(open(os.path.join(temp_dir,'postprocessor.py')).read(),globals())
      try:
          # clean up temp directory files for future runs
          os.remove(os.path.join(temp_dir,"postprocessor.py"))
      except:
          pass
      try:
          for i in pickle_file_list: 
              objectname=str(i)+".pkl"
              os.remove(os.path.join(temp_dir,objectname))
      except:
          pass
      return postprocessor

import os

def export_postprocessor(postprocessor_fxn,directory, globs=globals()):
    #postprocessor fxn should always be named "postprocessor" to work properly in aimodelshare process.
    try:
      import tempfile
      from zipfile import ZipFile
      import inspect
      import os

      globals().update(postprocessor_fxn.__globals__)

      folderpath=directory

      #create temporary folder
      temp_dir=tempfile.gettempdir()
      try:
          os.remove(os.path.join(folderpath,"postprocessor.zip"))
      except:
          pass
      #save function code within temporary folder
      source = inspect.getsource(postprocessor_fxn)
      with open(os.path.join(temp_dir,"postprocessor.py"), "w") as f:
          f.write(source)

      # create a ZipFile object
      zipObj = ZipFile(os.path.join(folderpath,"postprocessor.zip"), 'w')
      # Add postprocessor function to the zipfile
      zipObj.write(os.path.join(temp_dir,"postprocessor.py"),"postprocessor.py")

      #getting list of global variables used in function

      import inspect
      function_objects=list(inspect.getclosurevars(postprocessor_fxn).globals.keys())

      import sys
      modulenames = ["sklearn","keras","tensorflow","cv2","resize","pytorch"]
      function_objects_nomodules = [i for i in function_objects if i not in list(modulenames)]

      def savetopickle(function_objects_listelement):
        import pickle
        pickle.dump(globals()[function_objects_listelement], open( os.path.join(temp_dir,function_objects_listelement+".pkl"), "wb" ) )
        return function_objects_listelement

      savedpostprocessorobjectslist = list(map(savetopickle, function_objects_nomodules))

      # take savedpostprocessorobjectslist pkl files saved to tempdir to zipfile
      import pickle
      import string

      
      for i in savedpostprocessorobjectslist: 
          objectname=str(i)+".pkl"
          zipObj.write(os.path.join(temp_dir,objectname),objectname)

      # close the Zip File
      zipObj.close()

      try:
          # clean up temp directory files for future runs
          os.remove(os.path.join(temp_dir,"postprocessor.py"))

          for i in savedpostprocessorobjectslist: 
              objectname=str(i)+".pkl"
              os.remove(os.path.join(temp_dir,objectname))
      except:
          pass

    except Exception as e:
        print(e)

def upload_postprocessor(postprocessor_path, client, bucket, model_id, model_version):

  try:

    
    # Check the postprocessor {{{
    if not os.path.exists(postprocessor_path):
        raise FileNotFoundError(
            f"The postprocessor file at {postprocessor_path} does not exist"
        )

    
    file_name = os.path.basename(postprocessor_path)
    file_name, file_ext = os.path.splitext(file_name)
    
    from zipfile import ZipFile
    dir_zip = postprocessor_path

    #zipObj = ZipFile(os.path.join("./postprocessor.zip"), 'a')
    #/Users/aishwarya/Downloads/aimodelshare-master
    client["client"].upload_file(dir_zip, bucket, model_id + "/runtime_postprocessor" + ".zip")
  except Exception as e:
    print(e)


     

__all__ = [
    import_postprocessor,
    export_postprocessor,
    upload_postprocessor,
    
]
