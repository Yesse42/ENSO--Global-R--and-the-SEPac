"Contains the results of a PLS regression."
struct PLSRegressor{TX, TY, TXW, TYW, TXS, TYS, TXL, TYL, TXM, TYM, TXSD, TYSD}
    X::TX
    Y::TY
    n_components::Int
    X_weights::TXW
    Y_weights::TYW
    X_scores::TXS
    Y_scores::TYS
    X_loadings::TXL
    Y_loadings::TYL
    X_means::TXM
    Y_means::TYM
    X_stds::TXSD
    Y_stds::TYSD
end

"Contains the results of a PLS regression without storing the original data."
struct PLSRegressorReduced{TXW, TYW, TXS, TYS, TXL, TYL, TXM, TYM, TXSD, TYSD}
    n_components::Int
    X_weights::TXW
    Y_weights::TYW
    X_scores::TXS
    Y_scores::TYS
    X_loadings::TXL
    Y_loadings::TYL
    X_means::TXM
    Y_means::TYM
    X_stds::TXSD
    Y_stds::TYSD
end

function reduce_pls_model(pls::PLSRegressor; copy::Bool = false)
    if copy
        return PLSRegressorReduced(pls.n_components,
                                  copy(pls.X_weights), copy(pls.Y_weights),
                                  copy(pls.X_scores), copy(pls.Y_scores),
                                  copy(pls.X_loadings), copy(pls.Y_loadings),
                                  copy(pls.X_means), copy(pls.Y_means),
                                  copy(pls.X_stds), copy(pls.Y_stds))
    else
        return PLSRegressorReduced(pls.n_components,
                                  pls.X_weights, pls.Y_weights,
                                  pls.X_scores, pls.Y_scores,
                                  pls.X_loadings, pls.Y_loadings,
                                  pls.X_means, pls.Y_means,
                                  pls.X_stds, pls.Y_stds)
    end
end

abstract type ToCopy end
struct NoCopy <: ToCopy end
struct MakeCopy <: ToCopy end