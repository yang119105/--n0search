###############################################################################
# 批量数据集处理（评估 D3 时同时输出 SVM / KNN / RF 准确率 + 评分函数消融）
# 流程：ANOVA → LSD → α–n0 (Pearson+Spearman) + mRMR → 5×5 CV
# 评分函数族：
#   score_acc      = Acc_svm
#   score_acc_rho  = Acc_svm - λ·mean|ρ|
#   score_acc_IXY  = Acc_svm + γ·I(X;Y)
#   score_acc_IXX  = Acc_svm - β·I(X;X)
#   score_full     = Acc_svm - λ·mean|ρ| + γ·I(X;Y) - β·I(X;X)
# 导出：
#   1) 每数据集的网格 grid（含三分类器 Acc 及各类 Score）
#   2) D3 最优特征清单（按 score_full 选）
#   3) D3 三分类器 5×5 CV 汇总（最终结果）
#   4) 评分函数消融表 + 消融柱状图
#   5) 总汇总表 ALL_DATASETS_D3_Summary.csv
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
library(randomForest); library(infotheo); library(ggplot2)

set.seed(2025)

# -------------------- 可调参数 --------------------
CFG <- list(
  transpose        = TRUE,        # TRUE: 先转置并把第一列移到最后作为标签
  nbins            = 10,           # 互信息分箱数
  fdr_threshold    = 0.05,         # ANOVA 经 BH-FDR 的阈值
  alpha_candidates = seq(0, 1, by = 0.1),
  n0_candidates_by = 10,
  n0_max           = 100,
  lambda           = 0.4,          # 冗余惩罚（|ρ|）
  gamma            = 0.2,          # I(X;Y) 奖励
  beta             = 0.2,          # I(X;X) 冗余惩罚
  cv_reps          = 5,            # 5×5 CV：重复次数
  cv_k             = NULL          # NULL 自动按最小类选择；否则固定 k（2~5）
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

# -------------------- 单数据集流程（D3 + 评分函数消融） --------------------
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
  
  # ---- Step3: α–n0 + mRMR -> 网格搜索 + 评分函数消融 ----
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
  # 五种评分函数
  grid_results$score_acc      <- NA_real_
  grid_results$score_acc_rho  <- NA_real_
  grid_results$score_acc_IXY  <- NA_real_
  grid_results$score_acc_IXX  <- NA_real_
  grid_results$score_full     <- NA_real_
  
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
    
    # 五种评分（都基于 SVM）
    score_acc     <- acc_svm
    score_acc_rho <- acc_svm - cfg$lambda * m_spear
    score_acc_IXY <- acc_svm + cfg$gamma  * mi_rel_subset
    score_acc_IXX <- acc_svm - cfg$beta   * mi_red_subset
    score_full    <- acc_svm - cfg$lambda * m_spear +
      cfg$gamma  * mi_rel_subset -
      cfg$beta   * mi_red_subset
    
    grid_results[r, c("svm_mean", "knn_mean", "rf_mean",
                      "mean_spear", "mi_rel", "mi_red",
                      "score_acc", "score_acc_rho",
                      "score_acc_IXY", "score_acc_IXX",
                      "score_full")] <-
      c(acc_svm, acc_knn, acc_rf,
        m_spear, mi_rel_subset, mi_red_subset,
        score_acc, score_acc_rho,
        score_acc_IXY, score_acc_IXX,
        score_full)
    
    cat(sprintf("alpha=%.2f n0=%3d → Acc[SVM]=%.4f |ρ|=%.4f Ixy=%.3f Ixx=%.3f Score_full=%.4f\n",
                alpha, n0, acc_svm, m_spear, mi_rel_subset, mi_red_subset, score_full))
  }
  
  # -------------------- 评分函数消融：挑各自最优点 --------------------
  best_full <- grid_results[which.max(grid_results$score_full), ]
  best_acc  <- grid_results[which.max(grid_results$score_acc), ]
  best_rho  <- grid_results[which.max(grid_results$score_acc_rho), ]
  best_IXY  <- grid_results[which.max(grid_results$score_acc_IXY), ]
  best_IXX  <- grid_results[which.max(grid_results$score_acc_IXX), ]
  
  # 供后续 D3 构造使用：仍然以完整评分为准
  best_alpha <- best_full$alpha
  best_n0    <- best_full$n0
  
  # —— 构造消融结果表：只关心 (alpha, n0, 特征数≈n0, SVM 准确率) ——  
  ablation_summary <- data.frame(
    Mode     = factor(c("Acc_only","Acc_Rho","Acc_IXY","Acc_IXX","Full"),
                      levels = c("Acc_only","Acc_Rho","Acc_IXY","Acc_IXX","Full")),
    Alpha    = c(best_acc$alpha,  best_rho$alpha,
                 best_IXY$alpha,  best_IXX$alpha,
                 best_full$alpha),
    n0       = c(best_acc$n0,     best_rho$n0,
                 best_IXY$n0,     best_IXX$n0,
                 best_full$n0),
    Feats_D3 = c(best_acc$n0,     best_rho$n0,
                 best_IXY$n0,     best_IXX$n0,
                 best_full$n0),
    Acc_SVM  = c(best_acc$svm_mean,  best_rho$svm_mean,
                 best_IXY$svm_mean,  best_IXX$svm_mean,
                 best_full$svm_mean),
    MeanAbsRho = c(best_acc$mean_spear,  best_rho$mean_spear,
                   best_IXY$mean_spear,  best_IXX$mean_spear,
                   best_full$mean_spear),
    IXY_Mean   = c(best_acc$mi_rel,  best_rho$mi_rel,
                   best_IXY$mi_rel,  best_IXX$mi_rel,
                   best_full$mi_rel),
    IXX_Mean   = c(best_acc$mi_red,  best_rho$mi_red,
                   best_IXY$mi_red,  best_IXX$mi_red,
                   best_full$mi_red)
  )
  
  # -------------------- 根据 full-score 构造最终 D3 并做 5×5 CV --------------------
  abs_corr_best <- best_alpha * abs_pear + (1 - best_alpha) * abs_spear
  diag(abs_corr_best) <- 0
  sel_best <- greedy_select(abs_corr_best, best_n0)
  D3_idx_in_all <- if (length(sel_best)) D2_idx_in_all[sel_best] else integer(0)
  features_D3 <- if (length(D3_idx_in_all)) features_all[, D3_idx_in_all, drop=FALSE] else features_all
  
  cv_best <- run_5x5_cv_all(features_D3, labels, reps = cfg$cv_reps, k = cfg$cv_k)
  
  # -------------------- 导出各类结果 --------------------
  bn <- tools::file_path_sans_ext(basename(file_path))
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  # 1) 网格结果（包含所有 Score）
  write.csv(grid_results,
            file.path(out_dir, sprintf("%s_alpha_n0_grid_results.csv", bn)),
            row.names = FALSE)
  
  # 2) D3 特征清单（Full 评分下的最优子集）
  write.csv(data.frame(Feature = colnames(features_D3)),
            file.path(out_dir, sprintf("%s_D3_optimal_features.csv", bn)),
            row.names = FALSE)
  
  # 3) 评分函数消融表
  write.csv(ablation_summary,
            file.path(out_dir, sprintf("%s_D3_ScoreAblation.csv", bn)),
            row.names = FALSE)
  
  # 4) 评分函数消融可视化（SVM 准确率对比）
  p_abl <- ggplot(ablation_summary,
                  aes(x = Mode, y = Acc_SVM, fill = Mode)) +
    geom_bar(stat = "identity", width = 0.6) +
    geom_text(aes(label = sprintf("%.3f", Acc_SVM)),
              vjust = -0.5, size = 4) +
    ylim(0, 1.05) +
    labs(title = paste("Score 消融对比 -", bn),
         x = "评分策略",
         y = "SVM 5×5 CV 准确率") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5),
          legend.position = "none")
  
  ggsave(file.path(out_dir, sprintf("%s_D3_ScoreAblation.png", bn)),
         plot = p_abl, width = 7, height = 5, dpi = 300)
  
  # 5) D3 最终汇总（供 ALL_DATASETS_D3_Summary 使用）
  d3_summary <- data.frame(
    Dataset       = bn,
    Samples       = n_samples,
    Feats_All     = ncol(features_all),
    Feats_D1      = ncol(features_D1),
    Feats_D2      = ncol(features_D2),
    Feats_D3      = ncol(features_D3),
    Alpha_Best    = best_alpha,
    n0_Best       = best_n0,
    Acc_SVM_Mean  = cv_best$svm_mean, Acc_SVM_SD = cv_best$svm_sd,
    Acc_KNN_Mean  = cv_best$knn_mean, Acc_KNN_SD = cv_best$knn_sd,
    Acc_RF_Mean   = cv_best$rf_mean,  Acc_RF_SD  = cv_best$rf_sd,
    MeanAbsRho    = best_full$mean_spear,
    IXY_Mean      = best_full$mi_rel,
    IXX_Mean      = best_full$mi_red,
    Score_Best    = best_full$score_full,
    stringsAsFactors = FALSE
  )
  write.csv(d3_summary,
            file.path(out_dir, sprintf("%s_D3_cv_summary.csv", bn)),
            row.names = FALSE)
  
  list(
    dataset     = bn,
    grid        = grid_results,
    D3_features = colnames(features_D3),
    best        = d3_summary,
    ablation    = ablation_summary
  )
}

# -------------------- 主入口：批量处理 --------------------
files <- c(
  "C:/Users/PC/Desktop/stomach.csv"
 
)

out_dir <- "C:/Users/PC/Desktop/FS_Batch_Out"

all_results <- lapply(files, function(fp) process_one_dataset(fp, out_dir = out_dir, cfg = CFG))
summary_all <- dplyr::bind_rows(lapply(all_results, function(x) x$best))
print(summary_all)
write.csv(summary_all,
          file.path(out_dir, "ALL_DATASETS_D3_Summary.csv"),
          row.names = FALSE)

message("全部完成: ", out_dir)
