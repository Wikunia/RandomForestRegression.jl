using DataFrames, CSV, LinearAlgebra, RandomForestRegression, Statistics, StatsBase
using DecisionTree

RFR = RandomForestRegression

file = "./data/train.csv"
df = CSV.File(file) |> DataFrame!

feature_cols = [:LotFrontage, :LotArea, :OverallQual, :OverallCond, :YearBuilt, :TotalBsmtSF, Symbol("1stFlrSF"), 
                Symbol("2ndFlrSF"), :BedroomAbvGr, :KitchenAbvGr, :TotRmsAbvGrd, :GarageCars, :PoolArea, :MiscVal, :GrLivArea,
                :LowQualFinSF, :GarageYrBlt, :GarageArea
                ]

pred_col = :SalePrice
df = df[:,vcat(feature_cols,pred_col)]

function isNotValid(x, type)
    try 
        nx = parse(type, x)
        return false
    catch
        return true
    end
end

function make_valid_df!(df, feature_cols)
    for col in feature_cols
        if !isa(df[1,col], Int64) 
            invalid_rows = isNotValid.(df[:,col], Int64)
            valid_rows = (!).(invalid_rows)
            mean_val = string(convert(Int64,floor(mean(parse.(Int64,df[valid_rows,col])))))
            df[invalid_rows,col] = mean_val
            df[:,col] = parse.(Int64, df[:,col])
        end
    end
    return df
end

make_valid_df!(df, feature_cols)


df = disallowmissing!(df)

feature_matrix = convert(Array, transpose(convert(Matrix, df[:,feature_cols])))

train_ys = convert(Vector{Float64}, df[:,pred_col])

nof_real_train = convert(Int64, floor(0.8*length(train_ys)))
println("Train on ",nof_real_train, " of ", length(train_ys))
real_train_ys = train_ys[1:nof_real_train]
@views train_feature_matrix = feature_matrix[:,1:nof_real_train]
cv_ys = train_ys[nof_real_train+1:end]
@views cv_feature_matrix = feature_matrix[:,nof_real_train+1:end]

forest = RFR.create_random_forest(train_feature_matrix, real_train_ys, 100)

pred_ys = RFR.predict_forest(forest, cv_feature_matrix; default=mean(real_train_ys))

mae = sum(abs.(pred_ys.-cv_ys))/length(pred_ys)
println("MAE: ", mae)
println("RMSD: ", rmsd(log.(pred_ys), log.(cv_ys)))

println("")
println("What if we just take the mean?")
pred_ys = mean(real_train_ys)*ones(length(cv_ys))
mae = sum(abs.(pred_ys.-cv_ys))/length(pred_ys)
println("MAE: ", mae)
println("RMSD: ", rmsd(log.(pred_ys), log.(cv_ys)))


test_file = "./data/test.csv"
test_df = CSV.File(test_file) |> DataFrame!

test_df = test_df[:,vcat(:Id,feature_cols)]
make_valid_df!(test_df, feature_cols)

test_df = disallowmissing!(test_df)

test_feature_matrix = convert(Array, transpose(convert(Matrix, test_df[:,feature_cols])))

pred_test_ys = RFR.predict_forest(forest, test_feature_matrix; default=mean(real_train_ys))

submission_df = DataFrame(Id=test_df[:Id], SalePrice=pred_test_ys)
CSV.write("data/submission_more_fts.csv", submission_df)


# julia package
feature_matrix = convert(Matrix, df[:,feature_cols])
nof_real_train = convert(Int64, floor(0.8*length(train_ys)))
println("Train on ",nof_real_train, " of ", length(train_ys))
real_train_ys = train_ys[1:nof_real_train]
train_feature_matrix = feature_matrix[1:nof_real_train,:]
cv_ys = train_ys[nof_real_train+1:end]
cv_feature_matrix = feature_matrix[nof_real_train+1:end,:]

model = build_forest(real_train_ys, train_feature_matrix, 8, 100)
pred_ys = apply_forest(model, cv_feature_matrix)

mae = sum(abs.(pred_ys.-cv_ys))/length(pred_ys)
println("With the RandomForests julia package: ")
println("MAE: ", mae)
println("RMSD: ", rmsd(log.(pred_ys), log.(cv_ys)))

test_feature_matrix = convert(Array, convert(Matrix, test_df[:,feature_cols]))

pred_test_ys = apply_forest(model, test_feature_matrix)

submission_df = DataFrame(Id=test_df[:Id], SalePrice=pred_test_ys)
CSV.write("data/submission_julia_pkg.csv", submission_df)