"""
Main script to run on ec2 instance
"""
import boto3
import os
import cv2
import numpy as np
import findspark
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from pyspark.ml.feature import PCA
from pyspark.sql.functions import monotonically_increasing_id
from io import StringIO
from pyspark import SparkConf


def param_s3():
    """
    Initialize S3 parameters
    :return:
    """
    bckt = 'opc-p8-data'
    reg = 'eu-west-3'
    id_access = ''
    secret_key = ''
    s3 = boto3.resource(
        service_name='s3',
        region_name=reg,
        aws_access_key_id=id_access,
        aws_secret_access_key=secret_key
    )
    os.environ["AWS_DEFAULT_REGION"] = reg
    os.environ["AWS_ACCESS_KEY_ID"] = id_access
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key
    s3_bucket = s3.Bucket(bckt)
    client = boto3.client('s3')
    return s3_bucket, client


def list_folders(client, bckt_name, prefix=''):
    """
    list all folders in given bucket and prefix
    :param client:
    :param bckt_name:
    :param prefix:
    """
    response = client.list_objects_v2(Bucket=bckt_name, Prefix=prefix, Delimiter='/')
    for content in response.get('CommonPrefixes', []):
        yield content.get('Prefix')


def find_images(path):
    """
    get all images in given s3 path
    :param path:
    :return:
    """
    list_files = []
    bckt, client = param_s3()
    for fldr in bckt.objects.filter(Prefix=path):
        list_files += [fldr.key]
    return list_files


def read_and_preprocess_img(path, target_size, bckt_name):
    """
    reading images and preprocessing
    :param path:
    :param target_size:
    :param bckt_name:
    :return:
    """
    bckt, client = param_s3()
    obj = client.get_object(Bucket=bckt_name, Key=path)
    img = obj['Body'].read()
    np_array = np.frombuffer(img, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    res = cv2.GaussianBlur(resized, (1, 1), 0)

    return res


def main(DEBUG=False):
    """
    Main function with DEBUG option for testing purposes
    :param DEBUG:
    """
    print('################ Initializing ################')
    bucket_name = 'opc-p8-data'
    access_id = ''
    access_key = ''
    output_folder = 'ML_FEATURES'
    training_set_folder = 'Training'
    main_folder_name = 'fruits-360-original-size'
    path_to_csv = f'{main_folder_name}/{output_folder}/ml_output.csv'
    n_batch = 500
    nb_samples_ = 200

    conf = (SparkConf().set(
        'spark.executor.extraJavaOptions',
        '-Dcom.amazonaws.services.s3.enableV4=true').set(
        'spark.driver.extraJavaOptions',
        '-Dcom.amazonaws.services.s3.enableV4=true'))

    findspark.init()

    bucket, s3_client = param_s3()
    sc = SparkContext(conf=conf)
    sc.setSystemProperty('com.amazonaws.services.s3.enableV4', 'true')
    hadoopConf = sc._jsc.hadoopConfiguration()
    hadoopConf.set("fs.s3a.awsAccessKeyId", access_id)
    hadoopConf.set("fs.s3a.awsSecretAccessKey", access_key)
    hadoopConf.set("fs.s3a.endpoint", "s3.eu-west-3.amazonaws.com")
    hadoopConf.set("com.amazonaws.services.s3a.enableV4", "true")
    hadoopConf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    hadoopConf.set('fs.s3a.aws.credentials.provider', 'org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider')

    spark = SparkSession(sc)
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    sc.setLogLevel("ERROR")

    print('################ Searching files ################')
    folder_list = list_folders(s3_client, bucket_name)
    fold = []
    for folder in folder_list:
        fold += [folder]
    if len(fold) > 1 or len(fold) == 0:
        folder = main_folder_name
    else:
        folder = fold[0]
    training_folder = folder + training_set_folder + '/'
    folder_list = list_folders(s3_client, bucket_name, prefix=training_folder)
    lst_all_folders = []
    for folder in folder_list:
        lst_all_folders += [folder]

    para = sc.parallelize(lst_all_folders)
    all_imgs = para.map(find_images).reduce(lambda a, b: a + b)

    print('################ Reading and preprocessing ################')
    max_idx = len(all_imgs)
    idx = 0
    all_img_in_folder = []
    nb = 0
    while idx < max_idx:
        print(f'Batch {nb}')
        if idx + n_batch < max_idx:
            para = sc.parallelize(all_imgs[idx:idx + n_batch])
        else:
            para = sc.parallelize(all_imgs[idx:max_idx])
        all_img_in_batch = para.map(lambda x: read_and_preprocess_img(x, (224, 224), bucket_name)).collect()
        all_img_in_folder += all_img_in_batch
        idx += n_batch
        nb += 1

    para = sc.parallelize(all_imgs)
    all_imgs_names = para.map(lambda x: [x.split('/')[-2]]).reduce(lambda a, b: a + b)

    print('################ Extracting features ################')
    model = VGG16()
    if DEBUG:
        nb_samples = nb_samples_
    else:
        nb_samples = len(all_imgs_names)
    img_names = spark.createDataFrame(
        np.array([all_imgs_names[:nb_samples]]).T.tolist(),
        ['fruitname']
    )
    x = np.asarray(all_img_in_folder[:nb_samples])
    preprocessed_img = preprocess_input(x)
    result = model.predict(preprocessed_img)

    spk_df_features = spark.createDataFrame(result.tolist())
    assembler = VectorAssembler(inputCols=spk_df_features.columns, outputCol='features')
    spk_df_features = assembler.transform(spk_df_features)

    print('################ Dimension reduction ################')
    final_nb_features = 0
    k_val = 400
    while final_nb_features == 0:
        pca = PCA(k=k_val, inputCol="features")

        model_pca = pca.fit(spk_df_features)

        model_pca.setOutputCol("output")
        cumsum_arr = model_pca.explainedVariance.cumsum()
        ideal_nb_features = np.argmax(cumsum_arr > 0.9)
        if ideal_nb_features > 0:
            final_nb_features = ideal_nb_features
        k_val += 100

    pca = PCA(k=final_nb_features, inputCol="features")
    model_pca = pca.fit(spk_df_features)
    model_pca.setOutputCol("output")
    spk_df_features_out = spark.createDataFrame(model_pca.transform(spk_df_features).collect())
    output_data = spk_df_features_out.select('output').collect()

    rdd = sc.parallelize(output_data)
    final_df = rdd.map(lambda x: x.output.toArray().tolist()).collect()
    output_dataframe = spark.createDataFrame(final_df)

    output_dataframe = output_dataframe.withColumn("id1", monotonically_increasing_id())
    img_names = img_names.withColumn("id2", monotonically_increasing_id())
    final_out_df = output_dataframe.join(img_names, output_dataframe.id1 == img_names.id2).drop("id1", "id2")

    print('################ Exporting to S3 ################')
    csv_buffer = StringIO()
    pd_df_out = final_out_df.toPandas()
    pd_df_out.to_csv(csv_buffer)
    bucket.Object(path_to_csv).put(Body=csv_buffer.getvalue())
    print('################ Job Done ################')


if __name__ == '__main__':
    main(DEBUG=True)
