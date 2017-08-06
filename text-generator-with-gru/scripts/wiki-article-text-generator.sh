#######################################
## Train wiki article text generator ##
#######################################

python scripts/train.py \
--modelName wiki_article_generator \
--textFilePath data/wiki.test.raw \
--removeNonASCII

####################################################
## Generate new wiki articles with text generator ##
####################################################

python scripts/generate_text.py --modelName wiki_article_generator