#プログラム1｜ライブラリの設定
import os

#プログラム2｜フォルダパスの取得
path = 'createFolder.txt'
fullpath = os.getcwd()

#プログラム3｜ファイル格納リスト
folderlist = []

#プログラム4｜テキストファイル内のデータをリストへ格納
with open(path, encoding='utf-8') as lines:
    for line in lines:
        line = line.replace('\n','')
        folderlist.append(line.split('\t'))

#プログラム5｜作成したいフォルダを1つずつ編集
for i, folders in enumerate(folderlist):
    list = []

    #プログラム6｜データがなければ、一つ前のパスを入れる
    for j, folder in enumerate(folders):
        if folder != '':
            list.append(folderlist[i][j])
        else:
            list.append(folderlist[i-1][j])
    folderlist[i] = list

    #プログラム7｜folderpathにフルパスを作る
    folderpath = fullpath +'/' + '/'.join(folderlist[i])
    
    # プログラム8｜同じ名前のフォルダが存在しなければ、フォルダを作成
    if os.path.exists(folderpath) == False:
        os.makedirs(folderpath)