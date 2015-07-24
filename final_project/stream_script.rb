require 'yaml'
require 'twitter'
require 'csv'

CREDENTIALS = YAML.load(File.read("./twitter_credentials.yml"))
PATH_TO_SF_TWEETS = "./data/all_sf_tweets_with_stream_2.csv"

client = Twitter::Streaming::Client.new do |config|
  config.consumer_key        = CREDENTIALS['consumer_key']
  config.consumer_secret     = CREDENTIALS['consumer_secret']
  config.access_token        = CREDENTIALS['access_token']
  config.access_token_secret = CREDENTIALS['access_token_secret']
end

CSV.open(PATH_TO_SF_TWEETS, 'ab') do |csv|
  client.filter(locations: '-122.75,36.8,-121.75,37.8') do |object|
    csv << [object.id, object.text, object.created_at, object.retweet_count, object.favorite_count] if object.is_a?(Twitter::Tweet)
  end
end
