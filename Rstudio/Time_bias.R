library(survival)
library(survminer)
library(dplyr)
library(readxl)

source("fn_utils.R")

#---------------------------------------
setwd("C://Users//mbsaad//Desktop//Other projects//Covid_vaccine//New_request//Github_tutorial//Rstudio")
df <- read_excel(("Covid_data_v2.xlsx"), sheet = "Main")
time_col <- 'RFS'
event_col <- 'RFS_status'

cols <-c('MRN','any_vax_3m_3m',time_col,event_col,'RT.End.Date','Vaccine_Date')
df_out <-df[cols]
names(df_out)[names(df_out) == time_col] <- "Time"
names(df_out)[names(df_out) == event_col] <- "Event"
names(df_out)[names(df_out) == "RT.End.Date"] <- "TX_EndDate"
names(df_out)[names(df_out) == "Vaccine_Date"] <- "Event_Date"
df_out$TX_EndDate <- as.Date(df_out$TX_EndDate)
df_out$Event_Date <- as.Date(df_out$Event_Date)
df_out$t_vax <- as.numeric(df_out$Event_Date - df_out$TX_EndDate) / 30

#---build TV table000
long_df <- build_timevarying_table(df_out)
#---run cox tv----
cox_tv <- coxph(Surv(start, stop, event) ~ vax_tv, data = long_df)
#summary(cox_tv)

# Extract results
cox_sum <- summary(cox_tv)
HR <- round(cox_sum$coef[1, "exp(coef)"], 2)
pval <- signif(cox_sum$coef[1, "Pr(>|z|)"], 3)
label_text <- paste0("HR = ", HR, "\nP = ", pval)

#---Simon-Makuch plots----
fit_sm <- survfit(Surv(start, stop, event) ~ vax_tv,data = long_df,id=id)

ggsurvplot(fit_sm,
           data        = long_df,
           palette     = c("red", "royalblue"),
           legend.labs = c("Not Vaccinated", "Vaccinated"),
           xlab        = "Time (months)",
           ylab        = "Survival probability",
           title       = "TV-cox RFS (Simon-Makuch)",
           xlim        = c(0, 60),
           break.x.by  = 12,
           risk.table  = FALSE,
           censor      = TRUE,
           ggtheme     = theme_classic())

# Add annotation
fit_sm$plot <- fit_sm$plot + annotate("text", x = 40, y = 0.2, label = label_text, size = 5)
  


