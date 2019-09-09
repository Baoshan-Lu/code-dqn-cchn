#coding=utf-8
#甄码农python代码
#使用zipfile做目录压缩，解压缩功能
import os,os.path
import zipfile
def zip_dir(dirname,zipfilename):
  filelist = []
  if os.path.isfile(dirname):
    filelist.append(dirname)
  else :
    for root, dirs, files in os.walk(dirname):
      for name in files:
        filelist.append(os.path.join(root, name))
  zf = zipfile.ZipFile(zipfilename, "w", zipfile.zlib.DEFLATED)
  for tar in filelist:
    arcname = tar[len(dirname):]
    #print arcname
    zf.write(tar,arcname)
  zf.close()

if __name__ == '__main__':
    path=r'2019-09-0210'
    zip_file=path+'-gpu.zip'
    zip_dir(path,zip_file)
    print('Successful Zip...')
#   unzip_file(r'E:/python/learning/zip.zip',r'E:/python/learning2')
