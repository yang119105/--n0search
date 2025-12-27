###############################################################################
# 批量数据集处理（评估 D3 时同时输出 SVM / KNN / RF 准确率 + 置换检验）
# 流程：ANOVA → LSD → α–n0 (Pearson+Spearman) + mRMR → 5×5 CV → Permutation Test
# 评分函数：Score = Acc_svm - λ·mean|ρ| + γ·I(X;Y) - β·I(X;X)
# 导出：每数据集的 grid（含三类器Acc）、D3特征清单、D3三类器CV汇总（含置换p值）、
#      置换分布文件，以及总汇总表 ALL_DATASETS_D3_Summary.csv
###############################################################################

suppressPackageStartupMessages({
  if(!require(e1071))         install.packages("e1071", dependencies=TRUE)
  if(!require(class))         install.packages("class", dependencies=TRUE)
  if(!require(caret))         install.packages("caret", dependencies=TRUE)
  if(!require(dplyr))         install.packages("dplyr", dependencies=TRUE)
  if(!require(ggplot2))       install.packages("ggplot2", dependencies=TRUE)
  if(!require(randomForest))  install.packages("randomForest", dependencies=TRUE)
  if(!require(infotheo))      install.packages("infotheo", dependencies=TRUE)
})

library(e1071); library(class); library(caret); library(dplyr)
library(randomForest); library(infotheo)

set.seed(2025)

# -------------------- 可调参数 --------------------
CFG <- list(
  transpose        = FALSE,        # TRUE: 先转置并把第一列移到最后作为标签
  nbins            = 10,           # 互信息分箱数
  fdr_threshold    = 0.05,         # ANOVA 经 BH-FDR 的阈值
  alpha_candidates = seq(0, 1, by = 0.1),
  n0_candidates_by = 10,
  n0_max           = 100,
  lambda           = 0.4,          # 冗余惩罚（|ρ|）
  gamma            = 0.2,          # I(X;Y) 奖励
  beta             = 0.2,          # I(X;X) 冗余惩罚
  cv_reps          = 5,            # 5×5 CV：重复次数
  cv_k             = NULL,         # NULL 自动按最小类选择；否则固定 k（2~5）
  perm_B           = 1000,          # 置换次数（建议≥200；稳健可设1000+）
  perm_seed        = 20251         # 置换检验随机种子
)

# -------------------- 工具函数 --------------------
safe_scale <- function(X_train, X_test) {
  mu  <- apply(X_train, 2, function(z) mean(z, na.rm=TRUE))
  sdv <- apply(X_train, 2, function(z) sd(z, na.rm=TRUE))
  sdv[!is.finite(sdv) | sdv == 0] <- 1
  list(
    train = scale(X_train, center = mu, scale = sdv),
    test  = scale(X_test,  center = mu, scale = sdv)
  )
}

# 稳健 5×5 CV：同时返回 SVM / KNN / RF 的 mean / sd（自动分层造折）
run_5x5_cv_all <- function(X_mat, y_vec, reps=5, k=NULL, seed=123) {
  set.seed(seed)
  y_vec <- droplevels(as.factor(y_vec))
  tab <- table(y_vec); min_cls <- min(tab)
  if (is.null(k)) k <- max(2, min(5, min_cls - 1))
  k <- max(2, k)
  
  acc_svm <- c(); acc_knn <- c(); acc_rf <- c()
  folds_all <- replicate(reps, createFolds(y_vec, k = k), simplify = FALSE)
  
  for (folds in folds_all) {
    for (test_idx in folds) {
      train_idx <- setdiff(seq_along(y_vec), test_idx)
      
      y_train <- droplevels(y_vec[train_idx])
      y_test  <- droplevels(y_vec[test_idx])
      if (length(unique(y_train)) < 2) next
      y_test <- fact
      or(y_test, levels = levels(y_train))
      
      X_train <- as.matrix(X_mat[train_idx, , drop = FALSE])
      X_test  <- as.matrix(X_mat[test_idx,  , drop = FALSE])
      
      sc <- safe_scale(X_train, X_test)
      X_train_s <- sc$train; X_test_s <- sc$test
      
      # SVM
      m_svm  <- e1071::svm(x = X_train_s, y = y_train, kernel = "linear", scale = FALSE)
      p_svm  <- predict(m_svm, X_test_s)
      acc_svm <- c(acc_svm, mean(as.character(p_svm) == as.character(y_test), na.rm = TRUE))
      
      # KNN
      k_knn  <- max(1, floor(sqrt(nrow(X_train_s))) + 1)
      p_knn  <- class::knn(train = X_train_s, test = X_test_s, cl = y_train, k = k_knn)
      acc_knn <- c(acc_knn, mean(as.character(p_knn) == as.character(y_test), na.rm = TRUE))
      
      # RF
      m_rf   <- randomForest::randomForest(x = X_train_s, y = y_train)
      p_rf   <- predict(m_rf, X_test_s)
      acc_rf <- c(acc_rf, mean(as.character(p_rf) == as.character(y_test), na.rm = TRUE))
    }
  }
  
  safe_mean <- function(v) if (length(v)) mean(v) else NA_real_
  safe_sd   <- function(v) if (length(v)) sd(v)   else NA_real_
  
  list(
    svm_mean = safe_mean(acc_svm), svm_sd = safe_sd(acc_svm),
    knn_mean = safe_mean(acc_knn), knn_sd = safe_sd(acc_knn),
    rf_mean  = safe_mean(acc_rf),  rf_sd  = safe_sd(acc_rf)
  )
}

greedy_select <- function(abs_corr, n0) {
  v <- ncol(abs_corr)
  if (is.null(v) || v == 0) return(integer(0))
  avg_corr_all <- apply(abs_corr, 2, function(z) mean(z, na.rm=TRUE))
  remaining <- seq_len(v)
  selected <- c(which.min(avg_corr_all))
  remaining <- setdiff(remaining, selected)
  while (length(selected) < n0 && length(remaining) > 0) {
    mean_corr_to_selected <- sapply(remaining, function(j) mean(abs_corr[j, selected], na.rm=TRUE))
    next_feat <- remaining[which.min(mean_corr_to_selected)]
    selected <- c(selected, next_feat)
    remaining <- setdiff(remaining, next_feat)
  }
  selected
}

#—— 固定折生成：返回长度=reps的列表；每个元素是createFolds的返回（长度=k）
make_cv_folds <- function(y_vec, reps=5, k=NULL) {
  y_vec <- droplevels(as.factor(y_vec))
  tab <- table(y_vec); min_cls <- min(tab)
  if (is.null(k)) k <- max(2, min(5, min_cls - 1))
  k <- max(2, k)
  replicate(reps, createFolds(y_vec, k = k), simplify = FALSE)
}

#—— 用“既定折”评估三分类器（与 run_5x5_cv_all 逻辑一致，但不再自行造折）
eval_cv_all_given_folds <- function(X_mat, y_vec, folds_all) {
  y_vec <- droplevels(as.factor(y_vec))
  acc_svm <- c(); acc_knn <- c(); acc_rf <- c()
  for (folds in folds_all) {
    for (test_idx in folds) {
      train_idx <- setdiff(seq_along(y_vec), test_idx)
      y_train <- droplevels(y_vec[train_idx])
      y_test  <- droplevels(y_vec[test_idx])
      if (length(unique(y_train)) < 2) next
      y_test <- factor(y_test, levels = levels(y_train))
      X_train <- as.matrix(X_mat[train_idx, , drop = FALSE])
      X_test  <- as.matrix(X_mat[test_idx,  , drop = FALSE])
      sc <- safe_scale(X_train, X_test)
      X_train_s <- sc$train; X_test_s <- sc$test
      # SVM
      m_svm  <- e1071::svm(x = X_train_s, y = y_train, kernel = "linear", scale = FALSE)
      p_svm  <- predict(m_svm, X_test_s)
      acc_svm <- c(acc_svm, mean(as.character(p_svm) == as.character(y_test), na.rm = TRUE))
      # KNN
      k_knn  <- max(1, floor(sqrt(nrow(X_train_s))) + 1)
      p_knn  <- class::knn(train = X_train_s, test = X_test_s, cl = y_train, k = k_knn)
      acc_knn <- c(acc_knn, mean(as.character(p_knn) == as.character(y_test), na.rm = TRUE))
      # RF
      m_rf   <- randomForest::randomForest(x = X_train_s, y = y_train)
      p_rf   <- predict(m_rf, X_test_s)
      acc_rf <- c(acc_rf, mean(as.character(p_rf) == as.character(y_test), na.rm = TRUE))
    }
  }
  safe_mean <- function(v) if (length(v)) mean(v) else NA_real_
  safe_sd   <- function(v) if (length(v)) sd(v)   else NA_real_
  list(
    svm_mean = safe_mean(acc_svm), svm_sd = safe_sd(acc_svm),
    knn_mean = safe_mean(acc_knn), knn_sd = safe_sd(acc_knn),
    rf_mean  = safe_mean(acc_rf),  rf_sd  = safe_sd(acc_rf)
  )
}

#—— 置换检验：复用既定折；返回p值与置换分布
perm_test_all <- function(X_mat, y_vec, folds_all, B=200, seed=20251) {
  set.seed(seed)
  # 观测值（再算一次，确保与null完全同折可比）
  obs <- eval_cv_all_given_folds(X_mat, y_vec, folds_all)
  
  null_svm <- numeric(B); null_knn <- numeric(B); null_rf <- numeric(B)
  for (b in seq_len(B)) {
    y_perm <- sample(y_vec)  # 随机打乱标签
    res_b  <- eval_cv_all_given_folds(X_mat, y_perm, folds_all)
    null_svm[b] <- res_b$svm_mean
    null_knn[b] <- res_b$knn_mean
    null_rf[b]  <- res_b$rf_mean
    if (b %% 50 == 0) cat(sprintf("  Perm %d/%d done.\n", b, B))
  }
  
  # 单侧经验p值（null ≥ obs）
  p_svm <- (1 + sum(null_svm >= obs$svm_mean, na.rm = TRUE)) / (1 + B)
  p_knn <- (1 + sum(null_knn >= obs$knn_mean, na.rm = TRUE)) / (1 + B)
  p_rf  <- (1 + sum(null_rf  >= obs$rf_mean,  na.rm = TRUE)) / (1 + B)
  
  list(
    obs = obs,
    p_values = c(SVM = p_svm, KNN = p_knn, RF = p_rf),
    null_means = data.frame(perm_id = 1:B, svm = null_svm, knn = null_knn, rf = null_rf)
  )
}

# -------------------- 单数据集流程（D3 评估三类器 + 置换检验） --------------------
process_one_dataset <- function(file_path, out_dir = getwd(), cfg = CFG) {
  message(">>> 处理数据集: ", file_path)
  dat_raw <- read.csv(file_path, header = FALSE, stringsAsFactors = FALSE)
  
  if (isTRUE(cfg$transpose)) {
    dat_raw <- as.data.frame(t(as.matrix(dat_raw)))
    dat_raw <- cbind(dat_raw[, -1, drop=FALSE], dat_raw[, 1, drop=FALSE]) # 第一列移到最后为标签
  }
  
  n_samples    <- nrow(dat_raw)
  n_total_cols <- ncol(dat_raw)
  labels <- as.factor(dat_raw[, n_total_cols])
  features_all <- as.data.frame(dat_raw[, -n_total_cols, drop=FALSE])
  colnames(features_all) <- paste0("V", seq_len(ncol(features_all)))
  message(sprintf("样本: %d, 特征: %d", n_samples, ncol(features_all)))
  
  # ---- Step1: ANOVA + BH-FDR -> D1 ----
  p_vals <- vapply(seq_len(ncol(features_all)), function(i) {
    y <- labels; x <- features_all[[i]]
    keep <- is.finite(x) & !is.na(y)
    if (sum(keep) < 3) return(1)
    m <- try(aov(x[keep] ~ y[keep]), silent = TRUE)
    if (inherits(m, "try-error")) return(1)
    pv <- try(summary(m)[[1]]$`Pr(>F)`[1], silent = TRUE)
    if (inherits(pv, "try-error") || !is.finite(pv)) 1 else pv
  }, numeric(1))
  p_adj <- p.adjust(p_vals, method = "BH")
  D1_idx <- which(p_adj <= cfg$fdr_threshold)
  features_D1 <- if (length(D1_idx)) features_all[, D1_idx, drop=FALSE] else features_all
  message("D1 保留特征数: ", ncol(features_D1))
  
  # ---- Step2: LSD -> D2 ----
  classes <- levels(labels); a <- length(classes)
  selected_D2_idx_rel <- c()
  for (j in seq_len(ncol(features_D1))) {
    vec <- features_D1[[j]]
    an <- try(aov(vec ~ labels), silent = TRUE); if (inherits(an, "try-error")) next
    SSE_j <- sum(residuals(an)^2)
    df_error <- n_samples - a; if (df_error <= 0) next
    MS_error <- SSE_j / df_error
    t_crit <- qt(1 - 0.001/2, df = df_error)
    group_means <- tapply(vec, labels, mean)
    group_ns    <- tapply(vec, labels, length)
    passed <- FALSE
    for (k in seq_len(a-1)) {
      for (l in (k+1):a) {
        mean_diff <- abs(group_means[k] - group_means[l])
        LSD_j <- t_crit * sqrt(MS_error * (1 / group_ns[k] + 1 / group_ns[l]))
        if (is.finite(LSD_j) && is.finite(mean_diff) && (mean_diff > LSD_j)) { passed <- TRUE; break }
      }
      if (passed) break
    }
    if (passed) selected_D2_idx_rel <- c(selected_D2_idx_rel, j)
  }
  D2_idx_in_all <- if (length(selected_D2_idx_rel)) D1_idx[selected_D2_idx_rel] else D1_idx
  features_D2   <- if (length(D2_idx_in_all)) features_all[, D2_idx_in_all, drop=FALSE] else features_all
  message("D2 保留特征数: ", ncol(features_D2))
  
  # ---- Step3: α–n0 + mRMR -> 最优 D3 ----
  pear  <- suppressWarnings(cor(as.matrix(features_D2), method = "pearson",  use = "pairwise.complete.obs"))
  spear <- suppressWarnings(cor(as.matrix(features_D2), method = "spearman", use = "pairwise.complete.obs"))
  abs_pear  <- abs(pear);  diag(abs_pear)  <- 0
  abs_spear <- abs(spear); diag(abs_spear) <- 0
  
  # 离散化 + 互信息
  disc_feats_D2 <- as.data.frame(lapply(features_D2, function(x) {
    x <- as.numeric(x); x[!is.finite(x)] <- median(x[is.finite(x)], na.rm=TRUE)
    out <- try(discretize(x, disc = "equalfreq", nbins = cfg$nbins), silent = TRUE)
    if (inherits(out, "try-error")) {
      q <- unique(quantile(x, probs = seq(0,1,length.out=cfg$nbins+1), na.rm=TRUE))
      cut(x, breaks = q, include.lowest = TRUE, labels = FALSE)
    } else out
  }))
  y_disc <- droplevels(as.factor(labels))
  
  mi_xy <- sapply(disc_feats_D2, function(x) {
    v <- try(mutinformation(x, y_disc), silent = TRUE)
    if (inherits(v, "try-error") || !is.finite(v)) 0 else v
  })
  if (max(mi_xy, na.rm = TRUE) > 0) mi_xy <- mi_xy / max(mi_xy, na.rm = TRUE)
  
  mi_xx_cache <- new.env(hash=TRUE, parent=emptyenv())
  get_mi_xx <- function(i, j) {
    key <- if (i < j) paste0(i,"_",j) else paste0(j,"_",i)
    if (!exists(key, envir = mi_xx_cache)) {
      val <- try(mutinformation(disc_feats_D2[[i]], disc_feats_D2[[j]]), silent = TRUE)
      if (inherits(val, "try-error") || !is.finite(val)) val <- 0
      assign(key, val, envir = mi_xx_cache)
    }
    get(key, envir = mi_xx_cache)
  }
  
  n0_candidates <- seq(10, min(cfg$n0_max, ncol(features_D2)), by = cfg$n0_candidates_by)
  grid_results <- expand.grid(alpha = cfg$alpha_candidates, n0 = n0_candidates)
  grid_results$svm_mean   <- NA_real_
  grid_results$knn_mean   <- NA_real_
  grid_results$rf_mean    <- NA_real_
  grid_results$mean_spear <- NA_real_
  grid_results$mi_rel     <- NA_real_
  grid_results$mi_red     <- NA_real_
  grid_results$score      <- NA_real_
  
  message("开始 α–n0 + mRMR 网格搜索，共 ", nrow(grid_results), " 组...")
  
  for (r in seq_len(nrow(grid_results))) {
    alpha <- grid_results$alpha[r]; n0 <- grid_results$n0[r]
    abs_corr_combined <- alpha * abs_pear + (1 - alpha) * abs_spear
    diag(abs_corr_combined) <- 0
    
    sel_local <- greedy_select(abs_corr_combined, n0)
    if (!length(sel_local)) next
    feats_local <- features_D2[, sel_local, drop=FALSE]
    
    # 冗余：|ρ| 均值
    if (ncol(feats_local) > 1) {
      sub_spear <- suppressWarnings(cor(as.matrix(feats_local), method = "spearman", use = "pairwise.complete.obs"))
      m_spear <- mean(abs(sub_spear[upper.tri(sub_spear)]), na.rm=TRUE)
    } else m_spear <- 0
    
    # 互信息：I(X;Y), I(X;X)
    mi_rel_subset <- mean(mi_xy[sel_local], na.rm = TRUE)
    if (length(sel_local) > 1) {
      pairs <- combn(sel_local, 2)
      mi_vals <- apply(pairs, 2, function(idx) get_mi_xx(idx[1], idx[2]))
      mi_red_subset <- mean(mi_vals, na.rm = TRUE)
    } else mi_red_subset <- 0
    
    # 同时评估三类器
    cv_res <- run_5x5_cv_all(feats_local, labels, reps = cfg$cv_reps, k = cfg$cv_k)
    acc_svm <- cv_res$svm_mean; acc_knn <- cv_res$knn_mean; acc_rf <- cv_res$rf_mean
    
    # 用 SVM 的均值打分（更稳健），但把 KNN/RF 一并记录
    score_val <- acc_svm - cfg$lambda * m_spear + cfg$gamma * mi_rel_subset - cfg$beta * mi_red_subset
    
    grid_results[r, c("svm_mean", "knn_mean", "rf_mean",
                      "mean_spear", "mi_rel", "mi_red", "score")] <-
      c(acc_svm, acc_knn, acc_rf, m_spear, mi_rel_subset, mi_red_subset, score_val)
    
    cat(sprintf("alpha=%.2f n0=%3d → Acc[SVM]=%.4f  Acc[KNN]=%.4f  Acc[RF]=%.4f |ρ|=%.4f Ixy=%.3f Ixx=%.3f score=%.4f\n",
                alpha, n0, acc_svm, acc_knn, acc_rf, m_spear, mi_rel_subset, mi_red_subset, score_val))
  }
  
  # 选择最优（按 score）
  best_row   <- grid_results[which.max(grid_results$score), ]
  best_alpha <- best_row$alpha; best_n0 <- best_row$n0
  
  abs_corr_best <- best_alpha * abs_pear + (1 - best_alpha) * abs_spear
  diag(abs_corr_best) <- 0
  sel_best <- greedy_select(abs_corr_best, best_n0)
  D3_idx_in_all <- if (length(sel_best)) D2_idx_in_all[sel_best] else integer(0)
  features_D3 <- if (length(D3_idx_in_all)) features_all[, D3_idx_in_all, drop=FALSE] else features_all
  
  # 最优 D3：再评一次，拿到 mean±sd（用于常规汇总）
  cv_best <- run_5x5_cv_all(features_D3, labels, reps = cfg$cv_reps, k = cfg$cv_k)
  
  # ===== 置换检验（对最优 D3 子集）=====
  perm_out <- NULL
  if (!is.null(cfg$perm_B) && cfg$perm_B > 0) {
    folds_fixed <- make_cv_folds(labels, reps = cfg$cv_reps, k = cfg$cv_k) # 固定一组折
    perm_out <- perm_test_all(as.matrix(features_D3), labels,
                              folds_all = folds_fixed,
                              B = cfg$perm_B, seed = cfg$perm_seed)
  }
  
  # -------------------- 导出 --------------------
  bn <- tools::file_path_sans_ext(basename(file_path))
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  write.csv(grid_results, file.path(out_dir, sprintf("%s_alpha_n0_grid_results.csv", bn)), row.names = FALSE)
  write.csv(data.frame(Feature = colnames(features_D3)),
            file.path(out_dir, sprintf("%s_D3_optimal_features.csv", bn)), row.names = FALSE)
  
  # 导出置换分布（若开启）
  if (!is.null(perm_out)) {
    write.csv(perm_out$null_means,
              file.path(out_dir, sprintf("%s_D3_perm_null.csv", bn)),
              row.names = FALSE)
  }
  
  p_svm <- if (!is.null(perm_out)) perm_out$p_values["SVM"] else NA_real_
  p_knn <- if (!is.null(perm_out)) perm_out$p_values["KNN"] else NA_real_
  p_rf  <- if (!is.null(perm_out)) perm_out$p_values["RF"]  else NA_real_
  
  d3_summary <- data.frame(
    Dataset       = bn,
    Samples       = n_samples,
    Feats_All     = ncol(features_all),
    Feats_D1      = ncol(features_D1),
    Feats_D2      = ncol(features_D2),
    Feats_D3      = ncol(features_D3),
    Alpha_Best    = best_alpha,
    n0_Best       = best_n0,
    # 三类器最终 D3 的 mean±sd
    Acc_SVM_Mean  = cv_best$svm_mean, Acc_SVM_SD = cv_best$svm_sd,
    Acc_KNN_Mean  = cv_best$knn_mean, Acc_KNN_SD = cv_best$knn_sd,
    Acc_RF_Mean   = cv_best$rf_mean,  Acc_RF_SD  = cv_best$rf_sd,
    # 在最优点的网格指标（用于复现）
    MeanAbsRho    = best_row$mean_spear,
    IXY_Mean      = best_row$mi_rel,
    IXX_Mean      = best_row$mi_red,
    Score_Best    = best_row$score,
    # 置换检验 p 值
    P_SVM         = p_svm,
    P_KNN         = p_knn,
    P_RF          = p_rf,
    stringsAsFactors = FALSE
  )
  write.csv(d3_summary, file.path(out_dir, sprintf("%s_D3_cv_summary.csv", bn)), row.names = FALSE)
  
  list(
    dataset     = bn,
    grid        = grid_results,
    D3_features = colnames(features_D3),
    best        = d3_summary
  )
}

# -------------------- 主入口：批量处理 --------------------
files <- c(
  "C:/Users/PC/Desktop/A.csv"
   ,"C:/Users/PC/Desktop/B.csv"
   ,"C:/Users/PC/Desktop/Lung.csv"
  ,"C:/Users/PC/Desktop/LAT_ALB-AML.csv"
  ,"C:/Users/PC/Desktop/Prostate_transposed.csv"
  
)

out_dir <- "C:/Users/PC/Desktop/FS_Batch_Out"

all_results <- lapply(files, function(fp) process_one_dataset(fp, out_dir = out_dir, cfg = CFG))
summary_all <- dplyr::bind_rows(lapply(all_results, function(x) x$best))
print(summary_all)
write.csv(summary_all, file.path(out_dir, "ALL_DATASETS_D3_Summary.csv"), row.names = FALSE)

message("全部完成", out_dir)
