library(ggplot2)
library(dplyr)
library(latex2exp)
library(rjson)
library(data.table)


config <- fromJSON(paste(readLines("config.json"), collapse=""))


df_relmap <- data.frame()

for(d_dim in 2**seq(14)) {

	dat_path <- sprintf("%s/%s/LSA/relmapping_LSA_%d_%d_%d_%d_%.1f_%d.dat", 
			    	config$DATA_PATH,
				config$CORPUS_NAME,
				config$MODELS$LSA$M_CONTEXT,
				d_dim, #config$MODELS$LSA$D_DIM,
				as.integer(config$MODELS$LSA$UseWordByDoc),
				as.integer(config$MODELS$LSA$UseWordByWord),
				config$MODELS$LSA$ALPHA,
				config$MODELS$LSA$K_NEGATIVE)
	
	
	df <- read.table(dat_path, sep=" ", header=T)
	DT <- data.table(df)
	setkey(DT, Relation)
	affinities <- DT[, list(MedianRank = median(Rank)), by=Relation]
	affinities <- data.frame(affinities)

	for (Relation in affinities$Relation) {
		df_relmap <- rbind(df_relmap,
				     data.frame(DIM=d_dim, Relation=Relation, MedianRank=affinities[affinities$Relation == Relation,]$MedianRank))
	}
}



plot_medianRank <- ggplot(df_relmap, aes(x=DIM)) + 
			geom_point(aes(y=MedianRank, shape=Relation, color=Relation), size=4, alpha=0.5) +
			geom_line(aes(y=MedianRank, color=Relation), size=1) +
			scale_x_continuous(name="Dimensionality", breaks = unique(df_relmap$DIM), trans="log2") +
			scale_y_continuous(name = "Median Rank") +
			theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), axis.line = element_line(colour = "black"))




#		scale_y_continuous(name = "Median Rank", breaks=round(seq(0, max(df_fas$MedianRank) + 1, 1)/10)*10, 
#			                   sec.axis=sec_axis(trans=~.*coeff, name="%First", breaks=round(seq( 0, 25, 1)) )) +
#			scale_x_continuous(name="Dimensionality", breaks = unique(df_fas$DIM), trans="log2") +
#			geom_hline(data = df_ref, aes(yintercept = yi, alpha=0.25), show.legend=F) +
#			guides(linetype = "none") + 
#			theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), axis.line = element_line(colour = "black"))
