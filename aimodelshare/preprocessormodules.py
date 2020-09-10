import os
import zipfile
import sys
import pickle



def import_preprocessor(fname):
    # Load preprocessor version to session as preprocessor() function
    import importlib
    new_module = __import__(fname)
    return new_module


import os
def export_preprocessor(preprocessor_fxn):
    #preprocessor fxn should always be named "preprocessor" to work properly in aimodelshare process.
    try:
        import inspect
        source = inspect.getsource(preprocessor_fxn)
        with open("./aimodelshare/python/my_preprocessor.py", "w") as f:
            f.write(source)
    except Exception as e:
        print(e)

def upload_preprocessor(preprocessor_path, client, bucket, model_id, model_version):
    # Check the preprocessor {{{
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(
            f"The preprocessor file at {preprocessor_path} does not exist"
        )

    file_name = os.path.basename(preprocessor_path)
    file_name, file_ext = os.path.splitext(file_name)


    # Upload the preprocessor {{{
    try:
        client["client"].upload_file(
            preprocessor_path, bucket, model_id + "/runtime_preprocessor" + file_ext
        )
        client["client"].upload_file(
            preprocessor_path, bucket, model_id + f"/preprocessor_{model_version}" + file_ext
        )
       
            
        if(file_ext =='.py'):
            prep  = import_preprocessor(file_name)
            response = export_preprocessor(prep)
            try:
            	with zipfile.ZipFile('./aimodelshare/python/preprocessor_mostrecent.zip', 'a') as z:
            		z.write('./aimodelshare/python/my_preprocessor.py',os.path.join('python','preprocessor.py'))
            except Exception as e:
            	print(e)
            
    except Exception as err:
        return err
    # }}}


__all__ = [
    import_preprocessor,
    export_preprocessor,
    upload_preprocessor,
]

