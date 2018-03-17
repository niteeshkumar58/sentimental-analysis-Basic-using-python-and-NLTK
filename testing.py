import sentiment_mod as s
#text1 = "The story is as silly as it can get . It holds no novelty and the screenplay is packed with cliched scenes which are very predictable and boring. Dialouges are ordinary. Viraj Bhatt performs well but even he can't make up for the poor story and cliched scenes. Monalisa is impressive . Awadhesh Mishra is excellent and delivers a fine performance . Maya Yadav does well and puts on a good show"
text1 = "kapil sir is a pretty cool person."
text2 = "kapil sir sings very good but he is not a professional singer "
text3 = "those person who don't like kapil sir's song is not a good listener"
print(s.sentiment(text1))
print(s.sentiment(text2))
print(s.sentiment(text3))