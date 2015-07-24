require 'yaml'
require 'twitter'
require 'csv'

CREDENTIALS = YAML.load(File.read("./twitter_credentials.yml"))

client = Twitter::REST::Client.new do |config|
  config.consumer_key        = CREDENTIALS['consumer_key']
  config.consumer_secret     = CREDENTIALS['consumer_secret']
  config.access_token        = CREDENTIALS['access_token']
  config.access_token_secret = CREDENTIALS['access_token_secret']
end

QUERY = ('a'..'z').to_a.join(' OR ')
PATH_TO_PARK_CSV = "./data/park_tweets.csv"

PARK_GEO_CODES = {
  fort_mason: "37.804993,-122.430224,.2mi",
  alamo_square: "37.776366,-122.434548,.2mi",
  dolores: "37.759933,-122.427027,.2mi",
  crissy: "37.803931,-122.464557,.2mi",
  washington_square: "37.800788,-122.410076,.2mi",
  golden_gate: "37.770049,-122.458978,.3mi"
}

CSV.open(PATH_TO_PARK_CSV, "ab") do |csv|
  PARK_GEO_CODES.each do |park_name, coords|
    sleep(5)
    client.search(QUERY, geocode: coords).take(1000).each do |tweet|
      csv << [tweet.id, tweet.text, tweet.created_at, tweet.retweet_count, tweet.favorite_count, park_name]
    end
    sleep(5)
  end
end
