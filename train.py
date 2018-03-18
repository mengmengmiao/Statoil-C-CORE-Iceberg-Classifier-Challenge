# Training Data
df_train = pd.read_json('../input/train.json') # this is a dataframe


Xtrain = get_scaled_imgs(df_train)
Ytrain = np.array(df_train['is_iceberg'])
df_train.inc_angle = df_train.inc_angle.replace('na',0)
idx_tr = np.where(df_train.inc_angle>0)

Ytrain = Ytrain[idx_tr[0]]
Xtrain = Xtrain[idx_tr[0],...]

Xtr_more = get_more_images(Xtrain) 
Ytr_more = np.concatenate((Ytrain,Ytrain,Ytrain))

# K fold CV training
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
for fold_n, (train, test) in enumerate(kfold.split(Xtr_more, Ytr_more)):
    print("FOLD nr: ", fold_n)
    model = get_model()
    
    MODEL_FILE = 'mdl_simple_k{}_wght.hdf5'.format(fold_n)
    batch_size = 32
    mcp_save = ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15, verbose=1, epsilon=1e-4, mode='min')

    # set the epochs to 30 before training on your GPU
    model.fit(Xtr_more[train], Ytr_more[train],
        batch_size=batch_size,
        epochs=1,
        verbose=1,
        validation_data=(Xtr_more[test], Ytr_more[test]),
        callbacks=[mcp_save, reduce_lr_loss])
    
    model.load_weights(filepath = MODEL_FILE)

    score = model.evaluate(Xtr_more[test], Ytr_more[test], verbose=1)
    print('\n Val score:', score[0])
    print('\n Val accuracy:', score[1])

    SUBMISSION = './result/simplenet/sub_simple_v1_{}.csv'.format(fold_n)

    df_test = pd.read_json('../input/test.json')
    df_test.inc_angle = df_test.inc_angle.replace('na',0)
    Xtest = (get_scaled_imgs(df_test))
    pred_test = model.predict(Xtest)

    submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})
    print(submission.head(10))

    submission.to_csv(SUBMISSION, index=False)
    print("submission saved")
